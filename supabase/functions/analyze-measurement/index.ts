// supabase/functions/analyze-measurement/index.ts - Measurement-specific CAD analysis
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-correlation-id',
  'Access-Control-Allow-Methods': 'POST, GET, OPTIONS'
};

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  const correlationId = req.headers.get('X-Correlation-ID') || crypto.randomUUID();
  const startTime = Date.now();

  console.log(JSON.stringify({
    timestamp: new Date().toISOString(),
    level: 'INFO',
    correlation_id: correlationId,
    message: 'Measurement analysis request received'
  }));

  try {
    const contentType = req.headers.get('content-type') || '';
    let fileName: string = '';
    let fileData: ArrayBuffer | null = null;

    if (contentType.includes('multipart/form-data')) {
      const formData = await req.formData();
      const file = formData.get('file') as File;
      
      if (!file) {
        return new Response(
          JSON.stringify({ 
            error: 'No file provided in form data',
            correlation_id: correlationId
          }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
      
      fileName = file.name;
      fileData = await file.arrayBuffer();
      
    } else if (contentType.includes('application/json')) {
      const body = await req.json();
      fileName = body.fileName || '';
      
      if (body.fileUrl) {
        const fileResponse = await fetch(body.fileUrl);
        fileData = await fileResponse.arrayBuffer();
      } else if (body.fileData) {
        const base64Data = body.fileData.split(',')[1] || body.fileData;
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        fileData = bytes.buffer;
      }
    }

    if (!fileData) {
      return new Response(
        JSON.stringify({ 
          error: 'No file data provided',
          correlation_id: correlationId
        }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Check file type
    const fileType = fileName.split('.').pop()?.toLowerCase() || '';
    const isSTEP = ['step', 'stp', 'iges', 'igs'].includes(fileType);

    if (!isSTEP) {
      return new Response(
        JSON.stringify({ 
          error: 'Only STEP/IGES files are supported',
          correlation_id: correlationId
        }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Call geometry service /analyze-measurement endpoint
    const geometryServiceUrl = Deno.env.get('GEOMETRY_SERVICE_URL');
    
    if (!geometryServiceUrl) {
      return new Response(
        JSON.stringify({ 
          error: 'Geometry service not configured',
          correlation_id: correlationId
        }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      correlation_id: correlationId,
      message: 'Calling geometry service /analyze-measurement-v2',
      context: { 
        serviceUrl: geometryServiceUrl,
        fileName: fileName,
        fileSize: fileData.byteLength
      }
    }));

    const serviceFormData = new FormData();
    serviceFormData.append('file', new Blob([fileData]), fileName);

    // Use the NEW v2 endpoint with proper edge extraction
    const serviceResponse = await fetch(`${geometryServiceUrl}/analyze-measurement-v2`, {
      method: 'POST',
      body: serviceFormData,
      headers: {
        'X-Correlation-ID': correlationId,
        'Accept': 'application/json'
      },
      signal: AbortSignal.timeout(120000) // 2 minutes timeout
    });

    if (!serviceResponse.ok) {
      const errorText = await serviceResponse.text();
      console.error(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'ERROR',
        correlation_id: correlationId,
        message: 'Geometry service error',
        context: { status: serviceResponse.status, error: errorText }
      }));
      
      return new Response(
        JSON.stringify({ 
          error: 'Geometry service error',
          details: errorText,
          correlation_id: correlationId
        }),
        { status: serviceResponse.status, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const measurementData = await serviceResponse.json();

    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      correlation_id: correlationId,
      message: 'Measurement analysis complete',
      context: { 
        processingTime: Date.now() - startTime,
        hasMeshData: !!measurementData.mesh_data,
        hasMeasurementData: !!measurementData.measurement_data,
        faceCount: measurementData.measurement_data?.faces?.length || 0,
        edgeCount: measurementData.measurement_data?.edges?.length || 0
      }
    }));

    // Transform edges to tagged_edges format for frontend compatibility
    const taggedEdges = transformToTaggedEdges(measurementData.measurement_data?.edges || []);

    return new Response(
      JSON.stringify({
        success: true,
        correlation_id: correlationId,
        processing_time_ms: Date.now() - startTime,
        mesh_data: measurementData.mesh_data,
        measurement_data: measurementData.measurement_data,
        tagged_edges: taggedEdges,
        metadata: measurementData.metadata
      }),
      { 
        status: 200, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'ERROR',
      correlation_id: correlationId,
      message: 'Measurement analysis failed',
      context: { 
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined
      }
    }));

    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        message: error instanceof Error ? error.message : 'Unknown error',
        correlation_id: correlationId
      }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});

/**
 * Transform backend edge data to tagged_edges format expected by frontend
 */
function transformToTaggedEdges(edges: any[]): any[] {
  if (!edges || !Array.isArray(edges)) return [];
  
  return edges.map((edge, index) => {
    const taggedEdge: any = {
      feature_id: edge.edge_id ?? index,
      start: edge.start_point || [0, 0, 0],
      end: edge.end_point || [0, 0, 0],
      type: mapEdgeType(edge.edge_type),
    };

    // Add measurements based on edge type
    if (edge.diameter !== null && edge.diameter !== undefined) {
      taggedEdge.diameter = edge.diameter;
    }
    if (edge.radius !== null && edge.radius !== undefined) {
      taggedEdge.radius = edge.radius;
    }
    if (edge.length_mm !== null && edge.length_mm !== undefined) {
      taggedEdge.length = edge.length_mm;
    }
    if (edge.center) {
      taggedEdge.center = edge.center;
    }

    return taggedEdge;
  });
}

/**
 * Map backend edge type to frontend type
 */
function mapEdgeType(backendType: string | undefined): 'line' | 'circle' | 'arc' | 'ellipse' | 'spline' {
  if (!backendType) return 'line';
  
  const type = backendType.toLowerCase();
  if (type === 'circle') return 'circle';
  if (type === 'arc') return 'arc';
  if (type === 'ellipse') return 'ellipse';
  if (type === 'spline' || type === 'bspline') return 'spline';
  return 'line';
}
