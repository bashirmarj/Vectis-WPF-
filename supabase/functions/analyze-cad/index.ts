// supabase/functions/analyze-cad/index.ts - BRepNet v2.0 Integration
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-correlation-id',
  'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, PUT, DELETE'
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
    message: 'CAD analysis request received',
    context: {
      method: req.method,
      content_type: req.headers.get('content-type')
    }
  }));

  try {
    let fileName: string = '';
    let fileSize: number = 0;
    let fileData: ArrayBuffer | null = null;
    let material: string | undefined;
    let quantity: number = 1;

    const contentType = req.headers.get('content-type') || '';

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
      fileSize = file.size;
      fileData = await file.arrayBuffer();
      material = (formData.get('material') as string) || 'Steel';
      quantity = parseInt(formData.get('quantity') as string) || 1;
      
    } else if (contentType.includes('application/json')) {
      const body = await req.json();
      fileName = body.fileName || '';
      fileSize = body.fileSize || 0;
      material = body.material;
      quantity = body.quantity || 1;
      
      if (body.fileUrl) {
        const fileResponse = await fetch(body.fileUrl);
        fileData = await fileResponse.arrayBuffer();
        fileSize = fileData.byteLength;
      } else if (body.fileData) {
        const base64Data = body.fileData.split(',')[1] || body.fileData;
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        fileData = bytes.buffer;
        fileSize = fileData.byteLength;
      }
    }

    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      correlation_id: correlationId,
      message: 'Parameters parsed successfully',
      context: { fileName, fileSize, material, quantity }
    }));

    // Determine file type
    const fileType = fileName.split('.').pop()?.toLowerCase() || '';
    const isSTEP = ['step', 'stp', 'iges', 'igs'].includes(fileType);

    // Initialize analysis result
    let analysisResult: any = {
      success: true,
      correlation_id: correlationId,
      file_name: fileName,
      file_size: fileSize,
      quantity: quantity,
      method: 'heuristic',
      confidence: 0.85,
    };

    // Call BRepNet geometry service v2.0 if file data available
    if (fileData && isSTEP) {
      const geometryServiceUrl = Deno.env.get('GEOMETRY_SERVICE_URL');
      
      if (geometryServiceUrl) {
        try {
          console.log(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'INFO',
            correlation_id: correlationId,
            message: 'Calling BRepNet service v2.0',
            context: { serviceUrl: geometryServiceUrl }
          }));

          const serviceFormData = new FormData();
          serviceFormData.append('file', new Blob([fileData]), fileName);

          const serviceResponse = await fetch(`${geometryServiceUrl}/analyze`, {
            method: 'POST',
            body: serviceFormData,
            headers: {
              'X-Correlation-ID': correlationId,
              'Accept': 'application/json'
            },
            signal: AbortSignal.timeout(30000) // 30 seconds
          });

          if (serviceResponse.ok) {
            const serviceResult = await serviceResponse.json();
            
            // Transform BRepNet features to frontend format
            const features = serviceResult.features || [];
            
            // DEBUG: Log raw features from geometry service
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'DEBUG',
              correlation_id: correlationId,
              message: 'Raw features from geometry service',
              context: {
                features_count: features.length,
                features_sample: features.slice(0, 3).map((f: any) => ({
                  type: f.type,
                  confidence: f.confidence,
                  face_ids: f.face_ids
                }))
              }
            }));
            
            const geometricFeatures = {
              instances: features.map((feature: any) => ({
                type: feature.type,
                subtype: feature.subtype,
                face_indices: feature.face_ids || [],
                confidence: feature.confidence || 0,
                parameters: feature.parameters || {},
                bottom_faces: feature.bottom_faces,
                ml_detected: feature.ml_detected
              })),
              num_features_detected: features.length,
              num_faces_analyzed: serviceResult.mesh_data?.face_count || 0,
              confidence_score: features.length > 0 
                ? features.reduce((sum: number, f: any) => sum + (f.confidence || 0), 0) / features.length
                : 0,
              inference_time_sec: (serviceResult.processing_time_ms || 0) / 1000,
              recognition_method: serviceResult.metadata?.recognition_method || 'BRepNet'
            };
            
            // DEBUG: Log transformed features
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'DEBUG',
              correlation_id: correlationId,
              message: 'Transformed features for frontend',
              context: {
                instances_count: geometricFeatures.instances.length,
                instances_sample: geometricFeatures.instances.slice(0, 3).map((inst: any) => ({
                  type: inst.type,
                  confidence: inst.confidence
                })),
                type_summary: geometricFeatures.instances.reduce((acc: any, inst: any) => {
                  acc[inst.type] = (acc[inst.type] || 0) + 1;
                  return acc;
                }, {})
              }
            }));

            // Generate feature summary by type
            const featureSummary = features.reduce((acc: any, feature: any) => {
              const key = feature.subtype || feature.type;
              acc[key] = (acc[key] || 0) + 1;
              return acc;
            }, {});

            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'INFO',
              correlation_id: correlationId,
              message: 'BRepNet v2.0 analysis complete',
              context: { 
                has_mesh: !!serviceResult.mesh_data,
                has_face_mapping: !!serviceResult.mesh_data?.face_mapping,
                has_features: !!serviceResult.features,
                feature_count: features.length,
                recognition_method: serviceResult.metadata?.recognition_method,
                model_version: serviceResult.metadata?.model_version,
                processing_time: serviceResult.processing_time_ms
              }
            }));

            analysisResult = {
              ...analysisResult,
              ...serviceResult,
              geometric_features: {
                ...geometricFeatures,
                feature_summary: featureSummary
              },
              method: serviceResult.metadata?.recognition_method || 'brepnet'
            };
          } else {
            let serviceError = 'Unknown error';
            try {
              const errorBody = await serviceResponse.json();
              serviceError = errorBody.error || JSON.stringify(errorBody);
            } catch {
              serviceError = `HTTP ${serviceResponse.status}`;
            }

            console.error(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'ERROR',
              correlation_id: correlationId,
              message: 'BRepNet service failed - using fallback',
              context: { error: serviceError }
            }));
          }
        } catch (serviceErr) {
          console.error(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'ERROR',
            correlation_id: correlationId,
            message: 'BRepNet service error - using fallback',
            context: { error: (serviceErr as Error).message }
          }));
        }
      }
    }

    // Perform basic heuristic analysis if no geometry service result
    if (!analysisResult.mesh_data && fileSize > 0) {
      const estimatedVolume = Math.pow(fileSize / 10000, 0.6);
      const estimatedComplexity = Math.min(10, Math.max(1, Math.floor(Math.log10(fileSize / 1000))));
      
      analysisResult.geometry = {
        estimated_volume_cm3: estimatedVolume,
        estimated_complexity: estimatedComplexity,
        bounding_box: [100, 100, 100]
      };
    }

    const processingTime = Date.now() - startTime;
    
    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      correlation_id: correlationId,
      message: 'Analysis complete',
      context: { 
        processingTime,
        method: analysisResult.method
      }
    }));

    return new Response(
      JSON.stringify({
        ...analysisResult,
        processing_time_ms: processingTime,
        timestamp: new Date().toISOString()
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
      message: 'Edge function error',
      context: { error: (error as Error).message, stack: (error as Error).stack?.split('\n').slice(0, 3) }
    }));

    return new Response(
      JSON.stringify({
        error: (error as Error).message,
        correlation_id: correlationId,
        timestamp: new Date().toISOString()
      }),
      { 
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    );
  }
});
