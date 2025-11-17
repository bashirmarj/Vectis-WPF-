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

    // Call geometry service if file data available
    if (fileData && isSTEP) {
      const geometryServiceUrl = Deno.env.get('GEOMETRY_SERVICE_URL');
      const recognitionMethod = Deno.env.get('RECOGNITION_METHOD') || 'brepnet';
      
      if (geometryServiceUrl) {
        try {
          // Determine endpoint based on recognition method
          const endpoint = recognitionMethod.toLowerCase() === 'aag' 
            ? '/analyze-aag'  // AAG pattern matching
            : '/analyze';     // BRepNet ML

          console.log(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'INFO',
            correlation_id: correlationId,
            message: `Calling geometry service (${recognitionMethod})`,
            context: { 
              serviceUrl: geometryServiceUrl,
              endpoint: endpoint,
              method: recognitionMethod
            }
          }));

          const serviceFormData = new FormData();
          serviceFormData.append('file', new Blob([fileData]), fileName);

          const serviceResponse = await fetch(`${geometryServiceUrl}${endpoint}`, {
            method: 'POST',
            body: serviceFormData,
            headers: {
              'X-Correlation-ID': correlationId,
              'Accept': 'application/json'
            },
            signal: AbortSignal.timeout(60000) // 60 seconds
          });

          if (serviceResponse.ok) {
            const serviceResult = await serviceResponse.json();
            
            // Store large mesh data separately to avoid memory limits
            let meshStorageUrl = null;
            if (serviceResult.mesh_data) {
              const supabase = createClient(
                Deno.env.get('SUPABASE_URL') ?? '',
                Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
              );
              
              const meshFileName = `mesh_${correlationId}.json`;
              const meshData = JSON.stringify(serviceResult.mesh_data);
              
              const { data: uploadData, error: uploadError } = await supabase.storage
                .from('cad-files')
                .upload(`meshes/${meshFileName}`, meshData, {
                  contentType: 'application/json',
                  upsert: true
                });
              
              if (!uploadError && uploadData) {
                const { data: urlData } = supabase.storage
                  .from('cad-files')
                  .getPublicUrl(`meshes/${meshFileName}`);
                meshStorageUrl = urlData.publicUrl;
                
                console.log(JSON.stringify({
                  timestamp: new Date().toISOString(),
                  level: 'INFO',
                  correlation_id: correlationId,
                  message: 'Mesh data stored separately',
                  context: { mesh_url: meshStorageUrl, size_bytes: meshData.length }
                }));
              }
              
              // Remove mesh_data from result to save memory
              delete serviceResult.mesh_data;
            }
            
            // ENHANCED: Detailed logging of raw geometry service response
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'DEBUG',
              correlation_id: correlationId,
              message: 'Geometry service raw response',
              context: {
                has_features: !!serviceResult.features,
                features_count: serviceResult.features?.length || 0,
                features_sample: serviceResult.features?.slice(0, 2),
                mesh_stored_separately: !!meshStorageUrl,
                response_keys: Object.keys(serviceResult)
              }
            }));
            
            // 🔍 DEBUG POINT 1: Log RAW response from Python service
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'DEBUG',
              correlation_id: correlationId,
              message: '🔍 DEBUG 1: RAW Python service response',
              context: {
                response_keys: Object.keys(serviceResult),
                has_mesh_data_key: 'mesh_data' in serviceResult,
                mesh_data_type: typeof serviceResult.mesh_data,
                mesh_data_is_null: serviceResult.mesh_data === null,
                mesh_data_keys: serviceResult.mesh_data ? Object.keys(serviceResult.mesh_data) : null,
                vertices_exists: !!serviceResult.mesh_data?.vertices,
                vertices_type: typeof serviceResult.mesh_data?.vertices,
                vertices_length: serviceResult.mesh_data?.vertices?.length,
                indices_exists: !!serviceResult.mesh_data?.indices,
                indices_length: serviceResult.mesh_data?.indices?.length,
                normals_exists: !!serviceResult.mesh_data?.normals,
                normals_length: serviceResult.mesh_data?.normals?.length
              }
            }));
            
            // Transform BRepNet features to frontend format
            const features = serviceResult.features || [];
            
            // WARNING: Log if no features detected
            if (!serviceResult.features || features.length === 0) {
              console.warn(JSON.stringify({
                timestamp: new Date().toISOString(),
                level: 'WARN',
                correlation_id: correlationId,
                message: 'Geometry service returned NO features',
                context: {
                  file_name: fileName,
                  file_size: fileSize,
                  response_structure: Object.keys(serviceResult),
                  mesh_data_available: !!serviceResult.mesh_data
                }
              }));
            }
            
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
                has_vertex_face_ids: !!serviceResult.mesh_data?.vertex_face_ids,
                vertex_face_ids_length: serviceResult.mesh_data?.vertex_face_ids?.length,
                unique_face_ids: serviceResult.mesh_data?.vertex_face_ids 
                  ? [...new Set(serviceResult.mesh_data.vertex_face_ids.filter((id: number) => id !== -1))].length 
                  : 0,
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
              mesh_url: meshStorageUrl, // Reference to stored mesh data
              // ✅ EXPLICITLY include mesh_data from geometry service
              mesh_data: serviceResult.mesh_data || null,
              // ✅ ALSO include as 'geometry' for frontend compatibility
              geometry: serviceResult.mesh_data ? {
                vertices: serviceResult.mesh_data.vertices || [],
                indices: serviceResult.mesh_data.indices || [],
                normals: serviceResult.mesh_data.normals || [],
                vertex_face_ids: serviceResult.mesh_data.vertex_face_ids || [],
                face_mapping: serviceResult.mesh_data.face_mapping || {},
                hasVertices: !!(serviceResult.mesh_data.vertices?.length),
                hasIndices: !!(serviceResult.mesh_data.indices?.length),
                hasNormals: !!(serviceResult.mesh_data.normals?.length)
              } : null,
              geometric_features: {
                ...geometricFeatures,
                feature_summary: featureSummary
              },
              method: serviceResult.metadata?.recognition_method || 'brepnet'
            };
            
            // 🔍 DEBUG POINT 2: Verify geometry object creation
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'DEBUG',
              correlation_id: correlationId,
              message: '🔍 DEBUG 2: Geometry object created',
              context: {
                geometry_created: !!analysisResult.geometry,
                geometry_is_null: analysisResult.geometry === null,
                hasVertices_flag: analysisResult.geometry?.hasVertices,
                hasIndices_flag: analysisResult.geometry?.hasIndices,
                hasNormals_flag: analysisResult.geometry?.hasNormals,
                vertices_count: analysisResult.geometry?.vertices?.length || 0,
                indices_count: analysisResult.geometry?.indices?.length || 0,
                normals_count: analysisResult.geometry?.normals?.length || 0
              }
            }));
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
          
          // MOCK FEATURES: Add test data if enabled for development
          const enableMockFeatures = Deno.env.get('ENABLE_MOCK_FEATURES') === 'true';
          if (enableMockFeatures) {
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'INFO',
              correlation_id: correlationId,
              message: 'Using mock features for testing'
            }));
            
            analysisResult.geometric_features = {
              instances: [
                {
                  type: 'hole',
                  subtype: 'through_hole',
                  face_indices: [0, 1, 2, 3],
                  confidence: 0.95,
                  parameters: { diameter: 10.0, depth: 50.0 }
                },
                {
                  type: 'pocket',
                  subtype: 'rectangular',
                  face_indices: [4, 5, 6, 7, 8, 9],
                  confidence: 0.88,
                  parameters: { width: 20.0, length: 30.0, depth: 15.0 }
                },
                {
                  type: 'boss',
                  subtype: 'cylindrical',
                  face_indices: [10, 11, 12],
                  confidence: 0.92,
                  parameters: { diameter: 15.0, height: 25.0 }
                }
              ],
              num_features_detected: 3,
              num_faces_analyzed: 13,
              confidence_score: 0.917,
              inference_time_sec: 0.1,
              recognition_method: 'mock_fallback',
              feature_summary: {
                hole: 1,
                pocket: 1,
                boss: 1
              }
            };
          }
        }
      }
    }

    // Fallback heuristic analysis for non-STEP files or if no geometry service
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
    
    // 🔍 DEBUG POINT 3: Final response structure before sending
    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'DEBUG',
      correlation_id: correlationId,
      message: '🔍 DEBUG 3: Final response structure',
      context: {
        response_keys: Object.keys(analysisResult),
        has_mesh_data: 'mesh_data' in analysisResult,
        mesh_data_is_null: analysisResult.mesh_data === null,
        has_geometry: 'geometry' in analysisResult,
        geometry_is_null: analysisResult.geometry === null,
        geometry_hasVertices: analysisResult.geometry?.hasVertices,
        final_vertices_count: analysisResult.geometry?.vertices?.length || 0,
        final_indices_count: analysisResult.geometry?.indices?.length || 0,
        final_normals_count: analysisResult.geometry?.normals?.length || 0
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
