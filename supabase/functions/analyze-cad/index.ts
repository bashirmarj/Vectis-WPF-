// Supabase Edge Function: analyze-cad
// Integrates AAGNet multi-task feature recognition with existing CAD analysis pipeline

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-correlation-id',
};

// ✅ AAGNet feature class names (24 + stock)
const AAGNET_FEATURE_CLASSES = [
  'chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage',
  '6sides_passage', 'triangular_through_slot', 'rectangular_through_slot',
  'circular_through_slot', 'rectangular_through_step', '2sides_through_step',
  'slanted_through_step', 'Oring', 'blind_hole', 'triangular_pocket',
  'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
  'rectangular_blind_slot', 'v_circular_end_blind_slot',
  'h_circular_end_blind_slot', 'triangular_blind_step', 'circular_blind_step',
  'rectangular_blind_step', 'round', 'stock'
];

interface AAGNetFeatureInstance {
  type: string;
  face_indices: number[];
  bottom_faces: number[];
  confidence: number;
}

interface AAGNetResult {
  success: boolean;
  correlation_id: string;
  instances: AAGNetFeatureInstance[];
  semantic_labels: number[];
  extended_attributes: {
    face_attributes: number[][];
    edge_attributes: number[][];
  };
  num_faces: number;
  num_instances: number;
  processing_time: number;
}

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    const correlationId = req.headers.get('X-Correlation-ID') || crypto.randomUUID();
    const startTime = Date.now();

    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      correlation_id: correlationId,
      tier: 'EDGE',
      message: 'CAD analysis request received',
    }));

    // Parse request
    const { fileUrl, fileName, fileType } = await req.json();

    if (!fileUrl || !fileName) {
      return new Response(
        JSON.stringify({ error: 'Missing required parameters' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    // Determine processing path based on file type
    const isSTEP = ['step', 'stp', 'iges', 'igs'].includes(fileType.toLowerCase());
    
    let meshData: any = null;
    let mlFeatures: AAGNetResult | null = null;

    if (isSTEP) {
      // ✅ STEP/IGES files: Use AAGNet via Flask microservice
      console.log(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'INFO',
        correlation_id: correlationId,
        tier: 'EDGE',
        message: 'Processing STEP file with AAGNet',
        context: { fileName, fileType }
      }));

      try {
        // Call Flask microservice with AAGNet endpoint
        const flaskUrl = Deno.env.get('FLASK_SERVICE_URL') || 'http://localhost:5000';
        
        // Download file to temporary location
        const fileResponse = await fetch(fileUrl);
        const fileBlob = await fileResponse.blob();
        
        // Create FormData for Flask upload
        const formData = new FormData();
        formData.append('file', fileBlob, fileName);
        
        // Call AAGNet recognition endpoint
        const aagnetResponse = await fetch(`${flaskUrl}/api/aagnet/recognize`, {
          method: 'POST',
          body: formData,
          headers: {
            'X-Correlation-ID': correlationId
          }
        });

        if (!aagnetResponse.ok) {
          throw new Error(`AAGNet service error: ${aagnetResponse.status}`);
        }

        const aagnetResult: AAGNetResult = await aagnetResponse.json();

        console.log(JSON.stringify({
          timestamp: new Date().toISOString(),
          level: 'INFO',
          correlation_id: correlationId,
          tier: 'EDGE',
          message: 'AAGNet recognition complete',
          context: {
            num_instances: aagnetResult.num_instances,
            num_faces: aagnetResult.num_faces,
            processing_time: aagnetResult.processing_time
          }
        }));

        mlFeatures = aagnetResult;

        // Also get mesh data from Flask (tessellation)
        const meshResponse = await fetch(`${flaskUrl}/api/mesh/generate`, {
          method: 'POST',
          body: formData,
          headers: {
            'X-Correlation-ID': correlationId
          }
        });

        if (meshResponse.ok) {
          meshData = await meshResponse.json();
        }

      } catch (error: any) {
        console.error(JSON.stringify({
          timestamp: new Date().toISOString(),
          level: 'ERROR',
          correlation_id: correlationId,
          tier: 'EDGE',
          message: 'AAGNet processing failed',
          context: { error: error.message }
        }));

        // Fallback to basic mesh generation without ML features
        meshData = await generateBasicMesh(fileUrl, fileName, correlationId);
      }

    } else {
      // ✅ STL files: Basic tessellation (no AAGNet)
      console.log(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'INFO',
        correlation_id: correlationId,
        tier: 'EDGE',
        message: 'Processing STL file (basic tessellation)',
        context: { fileName, fileType }
      }));

      meshData = await generateBasicMesh(fileUrl, fileName, correlationId);
    }

    // Store mesh data in database
    let meshId: string | null = null;
    
    if (meshData) {
      const { data: insertedMesh, error: insertError } = await supabaseClient
        .from('cad_meshes')
        .insert({
          vertices: meshData.vertices,
          indices: meshData.indices,
          normals: meshData.normals,
          vertex_colors: meshData.vertex_colors,
          triangle_count: meshData.triangle_count,
          face_types: meshData.face_types,
          feature_edges: meshData.feature_edges,
          // ✅ NEW: AAGNet specific fields
          instance_features: mlFeatures ? {
            instances: mlFeatures.instances,
            num_instances: mlFeatures.num_instances,
            num_faces: mlFeatures.num_faces
          } : null,
          extended_attributes: mlFeatures?.extended_attributes || null,
          semantic_labels: mlFeatures?.semantic_labels || null,
          recognition_method: mlFeatures ? 'AAGNet' : 'tessellation',
          processing_time_ms: mlFeatures ? Math.round(mlFeatures.processing_time * 1000) : null
        })
        .select()
        .single();

      if (insertError) {
        console.error(JSON.stringify({
          timestamp: new Date().toISOString(),
          level: 'ERROR',
          correlation_id: correlationId,
          tier: 'EDGE',
          message: 'Database insertion failed',
          context: { error: insertError.message }
        }));
      } else {
        meshId = insertedMesh.id;
      }
    }

    // Generate feature summary for frontend
    const featureSummary = mlFeatures ? generateFeatureSummary(mlFeatures) : null;

    const response = {
      success: true,
      correlationId,
      meshId,
      volumeCm3: meshData?.volume_cm3,
      surfaceAreaCm2: meshData?.surface_area_cm2,
      complexityScore: meshData?.complexity_score,
      confidence: mlFeatures ? mlFeatures.instances.reduce((sum, f) => sum + f.confidence, 0) / mlFeatures.instances.length : null,
      method: mlFeatures ? 'AAGNet' : 'tessellation',
      mlFeatures: mlFeatures ? {
        instances: mlFeatures.instances,
        semantic_labels: mlFeatures.semantic_labels,
        num_faces: mlFeatures.num_faces,
        num_instances: mlFeatures.num_instances,
        processing_time: mlFeatures.processing_time
      } : null,
      featureSummary,
      recommendedProcesses: generateProcessRecommendations(mlFeatures),
      routingReasoning: generateRoutingReasoning(mlFeatures),
      machiningSummary: generateMachiningSummary(mlFeatures),
      processingTime: (Date.now() - startTime) / 1000
    };

    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      correlation_id: correlationId,
      tier: 'EDGE',
      message: 'CAD analysis complete',
      context: {
        total_time: response.processingTime,
        has_ml_features: !!mlFeatures,
        num_instances: mlFeatures?.num_instances || 0
      }
    }));

    return new Response(
      JSON.stringify(response),
      { 
        status: 200, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error: any) {
    console.error(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'ERROR',
      tier: 'EDGE',
      message: 'Edge function error',
      context: { error: error.message, stack: error.stack }
    }));

    return new Response(
      JSON.stringify({ 
        error: error.message,
        success: false 
      }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});

// Helper: Generate basic mesh without ML features
async function generateBasicMesh(fileUrl: string, fileName: string, correlationId: string) {
  // Placeholder - implement basic STL parsing or call Flask tessellation
  console.log(JSON.stringify({
    timestamp: new Date().toISOString(),
    level: 'WARN',
    correlation_id: correlationId,
    tier: 'EDGE',
    message: 'Basic mesh generation not fully implemented',
  }));
  
  return null;
}

// Helper: Generate feature summary for UI
function generateFeatureSummary(mlFeatures: AAGNetResult) {
  const typeCounts: Record<string, number> = {};
  
  mlFeatures.instances.forEach(inst => {
    typeCounts[inst.type] = (typeCounts[inst.type] || 0) + 1;
  });

  return {
    total_features: mlFeatures.num_instances,
    feature_types: typeCounts,
    high_confidence: mlFeatures.instances.filter(f => f.confidence >= 0.9).length,
    medium_confidence: mlFeatures.instances.filter(f => f.confidence >= 0.7 && f.confidence < 0.9).length,
    low_confidence: mlFeatures.instances.filter(f => f.confidence < 0.7).length,
    avg_confidence: mlFeatures.instances.reduce((sum, f) => sum + f.confidence, 0) / mlFeatures.instances.length
  };
}

// Helper: Generate process recommendations
function generateProcessRecommendations(mlFeatures: AAGNetResult | null): string[] {
  if (!mlFeatures) return ['CNC Milling'];

  const processes: Set<string> = new Set();
  
  mlFeatures.instances.forEach(inst => {
    if (inst.type.includes('hole')) {
      processes.add('Drilling');
    }
    if (inst.type.includes('pocket') || inst.type.includes('slot')) {
      processes.add('CNC Milling');
    }
    if (inst.type.includes('chamfer') || inst.type === 'round') {
      processes.add('Edge Finishing');
    }
  });

  return Array.from(processes);
}

// Helper: Generate routing reasoning
function generateRoutingReasoning(mlFeatures: AAGNetResult | null): string[] {
  if (!mlFeatures) return [];

  const reasoning: string[] = [];
  
  const holeCount = mlFeatures.instances.filter(f => f.type.includes('hole')).length;
  const pocketCount = mlFeatures.instances.filter(f => f.type.includes('pocket')).length;
  
  if (holeCount > 0) {
    reasoning.push(`${holeCount} hole feature(s) detected → Drilling operation required`);
  }
  if (pocketCount > 0) {
    reasoning.push(`${pocketCount} pocket feature(s) detected → Milling operation required`);
  }

  return reasoning;
}

// Helper: Generate machining summary
function generateMachiningSummary(mlFeatures: AAGNetResult | null) {
  if (!mlFeatures) return [];

  return mlFeatures.instances.map(inst => ({
    feature: inst.type,
    operation: getOperationForFeature(inst.type),
    complexity: inst.face_indices.length > 5 ? 'high' : 'medium',
    confidence: inst.confidence
  }));
}

// Helper: Map feature type to machining operation
function getOperationForFeature(featureType: string): string {
  if (featureType.includes('hole')) return 'Drilling';
  if (featureType.includes('pocket')) return 'Milling (Pocket)';
  if (featureType.includes('slot')) return 'Milling (Slot)';
  if (featureType.includes('step')) return 'Milling (Step)';
  if (featureType === 'chamfer' || featureType === 'round') return 'Edge Finishing';
  return 'Milling';
}
