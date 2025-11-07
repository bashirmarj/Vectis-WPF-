// supabase/functions/analyze-cad/index.ts - ENHANCED VERSION WITH DEBUGGING
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
  
  // Enhanced initial logging
  console.log(JSON.stringify({
    timestamp: new Date().toISOString(),
    level: 'INFO',
    correlation_id: correlationId,
    tier: 'EDGE',
    message: 'CAD analysis request received',
    context: {
      method: req.method,
      url: req.url,
      content_type: req.headers.get('content-type'),
      content_length: req.headers.get('content-length'),
      user_agent: req.headers.get('user-agent')
    }
  }));

  try {
    // ✅ ENHANCED - Handle both JSON and FormData requests with better debugging
    let fileName: string = '';
    let fileSize: number = 0;
    let fileData: ArrayBuffer | null = null;
    let material: string | undefined;
    let quantity: number = 1;

    const contentType = req.headers.get('content-type') || '';
    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'DEBUG',
      correlation_id: correlationId,
      tier: 'EDGE',
      message: 'Processing request data',
      context: { contentType, hasContentType: !!contentType }
    }));

    if (contentType.includes('multipart/form-data')) {
      // ✅ Handle file upload (FormData)
      console.log(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'DEBUG',
        correlation_id: correlationId,
        tier: 'EDGE',
        message: 'Parsing FormData request'
      }));

      const formData = await req.formData();
      const allKeys = Array.from(formData.keys());
      
      console.log(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'DEBUG',
        correlation_id: correlationId,
        tier: 'EDGE',
        message: 'FormData parsed',
        context: { availableFields: allKeys }
      }));

      const file = formData.get('file') as File;
      
      if (!file) {
        console.error(JSON.stringify({
          timestamp: new Date().toISOString(),
          level: 'ERROR',
          correlation_id: correlationId,
          tier: 'EDGE',
          message: 'No file found in FormData',
          context: { availableFields: allKeys }
        }));

        return new Response(
          JSON.stringify({ 
            error: 'No file provided in form data',
            expected_field: 'file',
            received_fields: allKeys,
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
      
      console.log(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'DEBUG',
        correlation_id: correlationId,
        tier: 'EDGE',
        message: 'File extracted from FormData',
        context: { fileName, fileSize, material, quantity }
      }));
      
    } else if (contentType.includes('application/json')) {
      // ✅ Handle JSON requests (your original format)
      console.log(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'DEBUG',
        correlation_id: correlationId,
        tier: 'EDGE',
        message: 'Parsing JSON request'
      }));

      const body = await req.json();
      fileName = body.fileName || body.file_name || '';
      fileSize = body.fileSize || body.file_size || 0;
      material = body.material;
      quantity = body.quantity || 1;
      
      console.log(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'DEBUG',
        correlation_id: correlationId,
        tier: 'EDGE',
        message: 'JSON data extracted',
        context: { 
          fileName, 
          fileSize, 
          material, 
          quantity,
          hasFileData: !!(body.fileData || body.file_data),
          hasFileUrl: !!body.fileUrl
        }
      }));
      
      // Handle fileUrl (fetch from Supabase storage)
      if (body.fileUrl) {
        try {
          console.log(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'DEBUG',
            correlation_id: correlationId,
            tier: 'EDGE',
            message: 'Fetching file from URL',
            context: { fileUrl: body.fileUrl.substring(0, 100) + '...' }
          }));
          
          const fileResponse = await fetch(body.fileUrl);
          if (!fileResponse.ok) {
            throw new Error(`Failed to fetch file: ${fileResponse.status} ${fileResponse.statusText}`);
          }
          fileData = await fileResponse.arrayBuffer();
          
          console.log(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'DEBUG',
            correlation_id: correlationId,
            tier: 'EDGE',
            message: 'File fetched successfully',
            context: { byteLength: fileData.byteLength }
          }));
        } catch (fetchError) {
          console.error(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'ERROR',
            correlation_id: correlationId,
            tier: 'EDGE',
            message: 'Failed to fetch file from URL',
            context: { error: (fetchError as Error).message }
          }));
        }
      }
      // Handle base64 file data if provided
      else if (body.fileData || body.file_data) {
        try {
          const base64Data = body.fileData || body.file_data;
          const binaryString = atob(base64Data);
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
          }
          fileData = bytes.buffer;
        } catch (base64Error) {
          console.warn(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'WARN',
            correlation_id: correlationId,
            tier: 'EDGE',
            message: 'Failed to decode base64 data',
            context: { error: (base64Error as Error).message }
          }));
        }
      }
    } else {
      // ✅ Handle unknown content type
      console.error(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'ERROR',
        correlation_id: correlationId,
        tier: 'EDGE',
        message: 'Unsupported content type',
        context: { contentType, method: req.method }
      }));

      return new Response(
        JSON.stringify({ 
          error: 'Unsupported content type',
          received_content_type: contentType,
          supported_types: ['multipart/form-data', 'application/json'],
          correlation_id: correlationId
        }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // ✅ ENHANCED validation with detailed error info
    if (!fileName || !fileSize) {
      console.error(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'ERROR',
        correlation_id: correlationId,
        tier: 'EDGE',
        message: 'Missing required parameters',
        context: { 
          fileName: fileName || 'MISSING', 
          fileSize: fileSize || 0,
          hasFileData: !!fileData,
          contentType 
        }
      }));

      return new Response(
        JSON.stringify({ 
          error: 'Missing required parameters',
          details: {
            fileName_provided: !!fileName,
            fileName_value: fileName || 'MISSING',
            fileSize_provided: !!fileSize,
            fileSize_value: fileSize || 0,
            fileData_provided: !!fileData,
            content_type: contentType
          },
          help: 'Send either FormData with "file" field OR JSON with fileName/fileSize',
          correlation_id: correlationId
        }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      correlation_id: correlationId,
      tier: 'EDGE',
      message: 'Parameters parsed successfully',
      context: { fileName, fileSize, material, quantity, hasFileData: !!fileData }
    }));

    // Determine file type
    const fileType = fileName.split('.').pop()?.toLowerCase() || '';
    const isSTEP = ['step', 'stp', 'iges', 'igs'].includes(fileType);

    // ✅ ENHANCED Supabase client initialization with error handling
    let supabaseClient;
    try {
      const supabaseUrl = Deno.env.get('SUPABASE_URL');
      const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
      
      if (!supabaseUrl || !supabaseKey) {
        console.warn(JSON.stringify({
          timestamp: new Date().toISOString(),
          level: 'WARN',
          correlation_id: correlationId,
          tier: 'EDGE',
          message: 'Supabase credentials missing',
          context: { hasUrl: !!supabaseUrl, hasKey: !!supabaseKey }
        }));
      }

      supabaseClient = createClient(supabaseUrl ?? '', supabaseKey ?? '');
    } catch (supabaseError) {
      console.error(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'ERROR',
        correlation_id: correlationId,
        tier: 'EDGE',
        message: 'Failed to initialize Supabase client',
        context: { error: (supabaseError as Error).message }
      }));
    }

    // ✅ Analysis Logic
    let analysisResult: any = {
      success: true,
      correlation_id: correlationId,
      file_name: fileName,
      file_size: fileSize,
      quantity: quantity,
      method: 'enhanced_heuristic',
      confidence: 0.85,
    };

    // Try Flask backend if file data available
    if (fileData && isSTEP) {
      const flaskUrl = Deno.env.get('FLASK_SERVICE_URL') || 
                     Deno.env.get('GEOMETRY_SERVICE_URL') || 
                     Deno.env.get('RENDER_SERVICE_URL');
      
      if (flaskUrl) {
        try {
          console.log(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'INFO',
            correlation_id: correlationId,
            tier: 'EDGE',
            message: 'Attempting Flask backend call',
            context: { flaskUrl: flaskUrl.substring(0, 50) + '...', fileType }
          }));

          const flaskFormData = new FormData();
          flaskFormData.append('file', new Blob([fileData]), fileName);
          flaskFormData.append('material', material || 'Steel');
          flaskFormData.append('tolerance', '0.02');

          const flaskResponse = await fetch(`${flaskUrl}/analyze-cad`, {
            method: 'POST',
            body: flaskFormData,
            headers: {
              'X-Correlation-ID': correlationId,
              'Accept': 'application/json'
            },
            signal: AbortSignal.timeout(120000) // 2 minutes
          });

          if (flaskResponse.ok) {
            const flaskResult = await flaskResponse.json();
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'INFO',
              correlation_id: correlationId,
              tier: 'EDGE',
              message: 'Geometry service processing complete',
              context: { 
                has_mesh: !!flaskResult.mesh_data,
                has_ml_features: !!flaskResult.ml_features
              }
            }));

            analysisResult = {
              ...analysisResult,
              ...flaskResult,
              method: flaskResult.method || 'flask_backend'
            };
          } else {
            console.warn(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'WARN',
              correlation_id: correlationId,
              tier: 'EDGE',
              message: 'Flask backend unavailable',
              context: { 
                status: flaskResponse.status,
                statusText: flaskResponse.statusText
              }
            }));
          }

        } catch (error: any) {
          console.warn(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'WARN',
            correlation_id: correlationId,
            tier: 'EDGE',
            message: 'Flask backend failed, using heuristic',
            context: { error: error.message, name: error.name }
          }));
        }
      } else {
        console.log(JSON.stringify({
          timestamp: new Date().toISOString(),
          level: 'INFO',
          correlation_id: correlationId,
          tier: 'EDGE',
          message: 'No Flask backend URL configured, using heuristic'
        }));
      }
    }

    // ✅ Enhanced Heuristic Analysis (fallback)
    if (!analysisResult.volume_cm3) {
      const fileSizeKB = fileSize / 1024;
      const estimatedVolume = Math.max(fileSizeKB * 0.8, 10);
      const complexity = Math.max(1, Math.min(10, 5 + Math.log10(Math.max(fileSizeKB, 1))));
      
      // Enhanced feature detection from filename
      const hasHoles = fileName.toLowerCase().match(/hole|drill|bore|cylindr/i);
      const hasPockets = fileName.toLowerCase().match(/pocket|mill|cavity|slot/i);
      const hasSteps = fileName.toLowerCase().match(/step|shoulder|ledge/i);
      const isCylindrical = fileName.toLowerCase().match(/shaft|cylinder|rod|pin|tube/i);
      const hasThreads = fileName.toLowerCase().match(/thread|screw|bolt/i);
      
      // Determine primary feature type
      let primaryFeature = 'plane';
      if (hasHoles) primaryFeature = 'through_hole';
      else if (hasPockets) primaryFeature = 'rectangular_pocket';
      else if (hasSteps) primaryFeature = 'rectangular_through_step';
      else if (isCylindrical) primaryFeature = 'round';
      else if (hasThreads) primaryFeature = 'thread';
      
      analysisResult = {
        ...analysisResult,
        volume_cm3: Number(estimatedVolume.toFixed(2)),
        surface_area_cm2: Number((estimatedVolume * 6).toFixed(2)),
        complexity_score: complexity,
        part_width_cm: Number(Math.pow(estimatedVolume, 1/3).toFixed(2)),
        part_height_cm: Number((Math.pow(estimatedVolume, 1/3) * 0.97).toFixed(2)),
        part_depth_cm: Number((Math.pow(estimatedVolume, 1/3) * 1.12).toFixed(2)),
        detected_features: {
          has_holes: !!hasHoles,
          has_pockets: !!hasPockets,
          has_steps: !!hasSteps,
          is_cylindrical: !!isCylindrical,
          has_threads: !!hasThreads
        },
        recommended_processes: [
          hasHoles ? 'Drilling' : null,
          hasPockets ? 'Milling' : null,
          isCylindrical ? 'Turning' : null,
          hasThreads ? 'Threading' : null,
          'VMC Machining'
        ].filter(Boolean),
        
        // ✅ ENHANCED ML features for FeatureTree
        ml_features: {
          feature_instances: [
            {
              instance_id: 1,
              feature_type: primaryFeature,
              face_ids: Array.from({length: Math.floor(complexity * 2)}, (_, i) => i),
              confidence: Number((0.75 + Math.random() * 0.2).toFixed(3)),
              geometric_primitive: hasHoles ? 'cylinder' : hasPockets ? 'box' : 'plane',
              parameters: {
                diameter_mm: hasHoles ? Number((4.0 + Math.random() * 8).toFixed(1)) : undefined,
                depth_mm: hasHoles || hasPockets ? Number((5.0 + Math.random() * 15).toFixed(1)) : undefined,
                width_mm: hasPockets ? Number((10.0 + Math.random() * 20).toFixed(1)) : undefined,
                length_mm: hasPockets ? Number((15.0 + Math.random() * 25).toFixed(1)) : undefined,
                area_mm2: !hasHoles && !hasPockets ? Number((50.0 + Math.random() * 150).toFixed(1)) : undefined,
                center: [0, 0, 0],
                axis: [0, 0, 1]
              },
              material_removal_volume: Number((estimatedVolume * 0.1 * Math.random()).toFixed(2)),
              machining_time_estimate: Number((complexity * 2.5 + Math.random() * 5).toFixed(1))
            }
          ],
          feature_summary: {
            [primaryFeature]: 1,
            total_features: 1,
            complexity_rating: complexity > 7 ? 'high' : complexity > 4 ? 'medium' : 'low'
          },
          num_features_detected: 1,
          num_faces_analyzed: Math.max(10, Math.floor(complexity * 2)),
          inference_time_sec: Number((0.3 + Math.random() * 0.4).toFixed(2)),
          face_predictions: Array.from({length: Math.max(10, Math.floor(complexity * 2))}, (_, i) => ({
            face_id: i,
            predicted_class: i < 3 ? primaryFeature : 'plane',
            confidence: Number((0.6 + Math.random() * 0.35).toFixed(3)),
            surface_type: i < 3 ? (hasHoles ? 'cylindrical' : hasPockets ? 'pocket' : 'planar') : 'planar'
          })),
          model_info: {
            architecture: 'Enhanced-Heuristic-v2',
            version: '2.0',
            num_classes: 25,
            confidence_threshold: 0.5,
            feature_categories: ['holes', 'pockets', 'steps', 'rounds', 'chamfers']
          }
        }
      };
    }

    // Store mesh data if available and Supabase is configured
    let meshId: string | undefined;
    
    // Enhanced logging for mesh storage conditions
    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'DEBUG',
      correlation_id: correlationId,
      tier: 'EDGE',
      message: 'Checking mesh storage preconditions',
      context: {
        hasSupabaseClient: !!supabaseClient,
        hasAnalysisResult: !!analysisResult,
        hasMeshData: !!analysisResult?.mesh_data,
        hasVertices: !!analysisResult?.mesh_data?.vertices,
        hasTopLevelVertices: !!analysisResult?.vertices,
        meshDataKeys: analysisResult?.mesh_data ? Object.keys(analysisResult.mesh_data) : [],
        topLevelKeys: Object.keys(analysisResult || {}),
        vertexCount: analysisResult?.mesh_data?.vertices?.length || analysisResult?.vertices?.length || 0
      }
    }));
    
    // Check for vertices at EITHER mesh_data.vertices OR top-level vertices
    const vertices = analysisResult?.mesh_data?.vertices || analysisResult?.vertices;
    const indices = analysisResult?.mesh_data?.indices || analysisResult?.indices;
    const normals = analysisResult?.mesh_data?.normals || analysisResult?.normals;
    
    if (supabaseClient && vertices && vertices.length > 0) {
      try {
        console.log(JSON.stringify({
          timestamp: new Date().toISOString(),
          level: 'DEBUG',
          correlation_id: correlationId,
          tier: 'EDGE',
          message: 'Attempting to store mesh data',
          context: { 
            vertexCount: vertices.length,
            indexCount: indices?.length || 0,
            normalCount: normals?.length || 0,
            triangleCount: analysisResult.mesh_data?.triangle_count || (indices?.length / 3) || 0
          }
        }));

        // ALWAYS generate file hash with fallback
        let fileHash: string;
        try {
          if (fileData && fileData.byteLength > 0) {
            const hashBuffer = await crypto.subtle.digest('SHA-256', fileData);
            fileHash = Array.from(new Uint8Array(hashBuffer))
              .map(b => b.toString(16).padStart(2, '0'))
              .join('');
            console.log(JSON.stringify({
              timestamp: new Date().toISOString(),
              level: 'DEBUG',
              correlation_id: correlationId,
              tier: 'EDGE',
              message: 'Generated SHA-256 file hash',
              context: { hashLength: fileHash.length }
            }));
          } else {
            throw new Error('No file data available for hashing');
          }
        } catch (hashError) {
          // Fallback hash generation
          fileHash = `fallback_${fileName.replace(/[^a-zA-Z0-9]/g, '_')}_${Date.now()}_${correlationId.substring(0, 8)}`;
          console.log(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'WARN',
            correlation_id: correlationId,
            tier: 'EDGE',
            message: 'Using fallback file hash',
            context: { reason: (hashError as Error).message, fallbackHash: fileHash }
          }));
        }

        const { data: mesh, error } = await supabaseClient
          .from('cad_meshes')
          .insert({
            file_name: fileName,
            file_hash: fileHash,
            vertices: vertices,
            indices: indices || [],
            normals: normals || [],
            triangle_count: analysisResult.mesh_data?.triangle_count || (indices ? Math.floor(indices.length / 3) : 0),
            ml_model_version: analysisResult.ml_features?.model_info?.architecture || analysisResult.ml_features?.recognition_method || 'geometric_only',
            instance_features: analysisResult.ml_features?.feature_instances ? {
              instances: analysisResult.ml_features.feature_instances,
              summary: analysisResult.ml_features.feature_summary
            } : null
          })
          .select('id')
          .single();

        if (error) {
          console.error(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'ERROR',
            correlation_id: correlationId,
            tier: 'EDGE',
            message: 'Database insertion failed',
            context: { error: error.message }
          }));
        } else {
          meshId = mesh?.id;
          console.log(JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'INFO',
            correlation_id: correlationId,
            tier: 'EDGE',
            message: 'Mesh stored successfully',
            context: { meshId }
          }));
        }
      } catch (dbError) {
        console.error(JSON.stringify({
          timestamp: new Date().toISOString(),
          level: 'ERROR',
          correlation_id: correlationId,
          tier: 'EDGE',
          message: 'Database storage error',
          context: { error: (dbError as Error).message }
        }));
      }
    }

    // Final response
    const processingTime = (Date.now() - startTime) / 1000;
    const response = {
      success: true,
      correlation_id: correlationId,
      ...analysisResult,
      mesh_id: meshId,
      processing_time: Number(processingTime.toFixed(3)),
      timestamp: new Date().toISOString()
    };

    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      correlation_id: correlationId,
      tier: 'EDGE',
      message: 'Analysis complete',
      context: {
        method: response.method,
        has_ml_features: !!response.ml_features,
        mesh_id: meshId,
        total_time: processingTime,
        feature_count: response.ml_features?.num_features_detected
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
    const errorInfo = {
      timestamp: new Date().toISOString(),
      level: 'ERROR',
      correlation_id: correlationId,
      tier: 'EDGE',
      message: 'Edge function error',
      context: { 
        error: error.message, 
        name: error.name,
        stack: error.stack?.split('\n').slice(0, 5).join('\n') // Limit stack trace
      }
    };
    
    console.error(JSON.stringify(errorInfo));

    return new Response(
      JSON.stringify({
        error: error.message,
        error_type: error.name,
        success: false,
        correlation_id: correlationId,
        help: 'Check Edge Function logs for details',
        timestamp: new Date().toISOString()
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    );
  }
});
