// supabase/functions/analyze-cad/index.ts - BRepNet v2.0 Integration
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-correlation-id',
  'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, PUT, DELETE'
};

// Validation comparison function
function compareWithGroundTruth(aagFeatures: any[], asGroundTruth: any, correlationId: string, extractionMetadata?: any) {
  const checks: any[] = [];
  
  // Parse AS ground truth structure - access features from correct nested path
  const body = asGroundTruth.parts?.[0]?.bodies?.[0];
  const asFeatures = body?.features || {};
  
  console.log('🔍 DEBUG: AS Ground Truth Features Structure:', {
    has_filletChains: !!asFeatures.filletChains,
    filletChains_type: typeof asFeatures.filletChains,
    filletChains_length: asFeatures.filletChains?.length,
    filletChains_sample: asFeatures.filletChains?.slice(0, 2)
  });
  
  const asHoles = asFeatures.holes || [];
  const asFillets = asFeatures.filletChains || [];
  const asChamfers = asFeatures.chamferChains || [];
  const asShoulders = asFeatures.shoulders || [];
  const asShafts = asFeatures.shafts || [];
  const asThreads = asFeatures.threads || [];
  
  // Count pockets from prismaticMilling configurations
  const asPrismaticMilling = asFeatures.prismaticMilling || [];
  let asPocketCount = 0;
  asPrismaticMilling.forEach((pm: any) => {
    asPocketCount += (pm.configurations || []).length;
  });
  
  // Count AAG features by type
  const aagCounts: Record<string, number> = {};
  aagFeatures.forEach(f => {
    const type = f.type || 'unknown';
    aagCounts[type] = (aagCounts[type] || 0) + 1;
  });
  
  // Normalize AAG feature types to AS categories
  const aagHoles = (aagCounts['through_hole'] || 0) + 
                   (aagCounts['blind_hole'] || 0) + 
                   (aagCounts['counterbore'] || 0) + 
                   (aagCounts['countersink'] || 0);
  const aagPockets = (aagCounts['rectangular_pocket'] || 0) + 
                     (aagCounts['irregular_pocket'] || 0) + 
                     (aagCounts['circular_end_pocket'] || 0);
  const aagFillets = aagCounts['fillet'] || 0;
  const aagChamfers = aagCounts['chamfer'] || 0;
  const aagSteps = (aagCounts['step'] || 0) + (aagCounts['shoulder'] || 0);
  const aagShafts = aagCounts['shaft'] || 0;
  
  // Structure checks
  checks.push({
    category: 'Structure',
    name: 'Features object exists',
    passed: !!body?.features,
    expected: 'object',
    actual: typeof body?.features
  });
  
  // Count comparisons
  const addCountCheck = (name: string, asCount: number, aagCount: number, tolerance = 0) => {
    const passed = Math.abs(asCount - aagCount) <= tolerance;
    checks.push({
      category: 'Counts',
      name: name,
      passed: passed,
      expected: asCount,
      actual: aagCount,
      deviation: Math.abs(asCount - aagCount)
    });
  };
  
  addCountCheck('Holes', asHoles.length, aagHoles);
  addCountCheck('Pockets', asPocketCount, aagPockets);
  addCountCheck('Fillets', asFillets.length, aagFillets);
  addCountCheck('Chamfers', asChamfers.length, aagChamfers);
  addCountCheck('Shoulders/Steps', asShoulders.length + asShafts.length, aagSteps + aagShafts);
  addCountCheck('Threads', asThreads.length, aagCounts['thread'] || 0);
  
  // Parameter checks (if available)
  // Check first hole diameter if both have holes
  if (asHoles.length > 0 && aagHoles > 0) {
    const asHole = asHoles[0];
    const aagHoleFeature = aagFeatures.find(f => 
      ['through_hole', 'blind_hole', 'counterbore', 'countersink'].includes(f.type)
    );
    
    if (asHole.bore?.diameter && aagHoleFeature?.parameters?.diameter) {
      const asD = asHole.bore.diameter;
      const aagD = aagHoleFeature.parameters.diameter;
      const tolerance = asD * 0.05; // 5% tolerance
      checks.push({
        category: 'Parameters',
        name: 'First hole diameter',
        passed: Math.abs(asD - aagD) <= tolerance,
        expected: asD.toFixed(2),
        actual: aagD.toFixed(2),
        deviation: Math.abs(asD - aagD).toFixed(2)
      });
    }
  }
  
  // Calculate summary
  const passed = checks.filter(c => c.passed).length;
  const total = checks.length;
  const passRate = total > 0 ? ((passed / total) * 100).toFixed(1) : '0.0';
  
  return {
    summary: {
      total_checks: total,
      passed: passed,
      failed: total - passed,
      pass_rate: passRate + '%'
    },
    checks: checks,
    metadata: {
      correlation_id: correlationId,
      timestamp: new Date().toISOString(),
      aag_feature_count: aagFeatures.length,
      as_feature_count: asHoles.length + asPocketCount + asFillets.length + asChamfers.length + asShoulders.length + asShafts.length
    },
    extraction_log: extractionMetadata || null
  };
}

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
    let validationMode: boolean = false;
    let asGroundTruth: any = null;

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
      
      // Debug: Log raw request body
      console.log(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'DEBUG',
        correlation_id: correlationId,
        message: '🔍 RAW REQUEST BODY',
        context: {
          body_keys: Object.keys(body),
          validation_mode_value: body.validation_mode,
          validation_mode_type: typeof body.validation_mode,
          has_as_ground_truth: !!body.as_ground_truth,
          as_ground_truth_keys: body.as_ground_truth ? Object.keys(body.as_ground_truth) : null
        }
      }));
      
      fileName = body.fileName || '';
      fileSize = body.fileSize || 0;
      material = body.material;
      quantity = body.quantity || 1;
      
      // Extract validation parameters - handle both boolean and string values
      validationMode = body.validation_mode === true || body.validation_mode === 'true';
      asGroundTruth = body.as_ground_truth || null;
      
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
      context: { 
        fileName, 
        fileSize, 
        material, 
        quantity,
        validationMode: validationMode || false,
        hasGroundTruth: !!asGroundTruth
      }
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
            
            // 🔍 DEBUG: Log raw response from Render.com BEFORE any processing
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'DEBUG',
              correlation_id: correlationId,
              message: '🔍 RAW RENDER.COM RESPONSE',
              context: {
                response_keys: Object.keys(serviceResult),
                has_mesh_data: 'mesh_data' in serviceResult,
                mesh_data_is_null: serviceResult.mesh_data === null,
                mesh_data_type: typeof serviceResult.mesh_data,
                mesh_data_keys: serviceResult.mesh_data ? Object.keys(serviceResult.mesh_data) : 'NULL',
                vertices_exists: !!serviceResult.mesh_data?.vertices,
                vertices_length: serviceResult.mesh_data?.vertices?.length || 0,
                indices_length: serviceResult.mesh_data?.indices?.length || 0,
                normals_length: serviceResult.mesh_data?.normals?.length || 0,
                response_size_estimate: JSON.stringify(serviceResult).length
              }
            }));
            
            // Store large mesh data separately to avoid memory limits
            let meshStorageUrl = null;
            let preservedMeshData = null; // ✅ PRESERVE mesh_data before deletion
            
            if (serviceResult.mesh_data) {
              // ✅ PRESERVE mesh_data BEFORE storing/deleting
              preservedMeshData = { ...serviceResult.mesh_data };
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
              
              // 🔍 DEBUG: Log before deleting mesh_data
              console.log(JSON.stringify({
                timestamp: new Date().toISOString(),
                level: 'DEBUG',
                correlation_id: correlationId,
                message: '🔍 MESH DATA STORED - about to delete from serviceResult',
                context: {
                  mesh_stored_at: meshStorageUrl,
                  preserved_mesh_vertices: preservedMeshData?.vertices?.length || 0,
                  will_delete_from_serviceResult: true
                }
              }));
              
              // Remove mesh_data from result to save memory (but we preserved it above)
              delete serviceResult.mesh_data;
            } else if (serviceResult.mesh_data === null) {
              console.warn(JSON.stringify({
                timestamp: new Date().toISOString(),
                level: 'WARN',
                correlation_id: correlationId,
                message: '⚠️ RENDER.COM RETURNED NULL MESH_DATA',
                context: {
                  file_name: fileName,
                  file_size: fileSize,
                  response_keys: Object.keys(serviceResult),
                  possible_timeout: fileSize > 10000000, // >10MB
                  recommendation: 'Check Render.com logs for timeout or memory errors'
                }
              }));
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

            // 🔍 DEBUG: Verify preserved mesh data
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'DEBUG',
              correlation_id: correlationId,
              message: '🔍 USING PRESERVED MESH DATA',
              context: {
                preserved_mesh_exists: !!preservedMeshData,
                preserved_vertices: preservedMeshData?.vertices?.length || 0,
                mesh_url_exists: !!meshStorageUrl
              }
            }));
            
            analysisResult = {
              ...analysisResult,
              ...serviceResult,
              mesh_url: meshStorageUrl, // Reference to stored mesh data
              // ✅ USE PRESERVED mesh_data (not serviceResult.mesh_data which was deleted)
              mesh_data: preservedMeshData || null,
              // ✅ ALSO include as 'geometry' for frontend compatibility
              geometry: preservedMeshData ? {
                vertices: preservedMeshData.vertices || [],
                indices: preservedMeshData.indices || [],
                normals: preservedMeshData.normals || [],
                vertex_face_ids: preservedMeshData.vertex_face_ids || [],
                face_mapping: preservedMeshData.face_mapping || {},
                hasVertices: !!(preservedMeshData.vertices?.length),
                hasIndices: !!(preservedMeshData.indices?.length),
                hasNormals: !!(preservedMeshData.normals?.length)
              } : null,
              geometric_features: {
                ...geometricFeatures,
                feature_summary: featureSummary
              },
              method: serviceResult.metadata?.recognition_method || 'brepnet'
            };
            
            // Run validation if validation mode enabled
            if (validationMode && asGroundTruth) {
              console.log(JSON.stringify({
                timestamp: new Date().toISOString(),
                level: 'INFO',
                correlation_id: correlationId,
                message: 'Running validation against ground truth',
                context: {
                  aag_features: features.length,
                  has_ground_truth: true
                }
              }));
              
              try {
                // Collect extraction metadata for the log including full backend logs
                const extractionMetadata = {
                  processing_time_ms: serviceResult.processing_time_ms,
                  recognition_method: serviceResult.metadata?.recognition_method || 'AAG Pattern Matching',
                  correlation_id: correlationId,
                  features_detected: features.length,
                  feature_types: serviceResult.metadata?.feature_summary || {},
                  service_response: {
                    status: serviceResult.status,
                    success: serviceResult.success
                  },
                  backend_logs: serviceResult.processing_logs || null,
                  detailed_stats: serviceResult.metadata || {}
                };
                
                const validationReport = compareWithGroundTruth(features, asGroundTruth, correlationId, extractionMetadata);
                analysisResult.validation_report = validationReport;
                
                console.log(JSON.stringify({
                  timestamp: new Date().toISOString(),
                  level: 'INFO',
                  correlation_id: correlationId,
                  message: 'Validation complete',
                  context: {
                    total_checks: validationReport.summary.total_checks,
                    passed: validationReport.summary.passed,
                    pass_rate: validationReport.summary.pass_rate
                  }
                }));
              } catch (validationError: any) {
                console.log(JSON.stringify({
                  timestamp: new Date().toISOString(),
                  level: 'ERROR',
                  correlation_id: correlationId,
                  message: 'Validation failed',
                  context: {
                    error: validationError?.message || 'Unknown error'
                  }
                }));
              }
            }
            
            // 🔍 DEBUG POINT 2: Verify final response structure
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'DEBUG',
              correlation_id: correlationId,
              message: '🔍 DEBUG 2: FINAL RESPONSE TO FRONTEND',
              context: {
                response_keys: Object.keys(analysisResult),
                has_mesh_data: 'mesh_data' in analysisResult,
                mesh_data_is_null: analysisResult.mesh_data === null,
                mesh_data_vertices: analysisResult.mesh_data?.vertices?.length || 0,
                geometry_created: !!analysisResult.geometry,
                geometry_vertices: analysisResult.geometry?.vertices?.length || 0,
                mesh_url: analysisResult.mesh_url,
                has_features: !!analysisResult.geometric_features,
                feature_count: analysisResult.geometric_features?.num_features_detected || 0
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
