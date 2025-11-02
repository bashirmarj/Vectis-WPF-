import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const authHeader = req.headers.get('Authorization');
    if (!authHeader) {
      throw new Error('No authorization header');
    }

    const { data: { user }, error: authError } = await supabase.auth.getUser(
      authHeader.replace('Bearer ', '')
    );

    if (authError || !user) {
      throw new Error('Unauthorized');
    }

    const { 
      dataset_name = 'MFCAD',
      epochs = 100,
      batch_size = 16,
      learning_rate = 0.001 
    } = await req.json();

    // Check if dataset is ready
    const { data: dataset, error: datasetError } = await supabase
      .from('training_datasets')
      .select('*')
      .eq('name', dataset_name)
      .single();

    if (datasetError || !dataset) {
      throw new Error(`Dataset ${dataset_name} not found`);
    }

    if (dataset.download_status !== 'completed') {
      throw new Error(`Dataset ${dataset_name} is not ready. Status: ${dataset.download_status}`);
    }

    // Create training job
    const model_version = `v${new Date().getTime()}`;
    const { data: job, error: jobError } = await supabase
      .from('training_jobs')
      .insert({
        started_by: user.id,
        status: 'pending',
        dataset_name,
        dataset_path: dataset.storage_path,
        model_version,
        epochs,
        batch_size,
        learning_rate,
        logs: [
          {
            timestamp: new Date().toISOString(),
            message: 'Training job created',
            level: 'info'
          }
        ]
      })
      .select()
      .single();

    if (jobError) {
      throw new Error(`Failed to create training job: ${jobError.message}`);
    }

    console.log(`Created training job: ${job.id}`);

    // In a production environment, you would trigger an actual training process here
    // This could be:
    // 1. A call to a separate Python training service
    // 2. A job queue system
    // 3. A cloud ML platform like AWS SageMaker or Google Vertex AI
    
    // For now, we'll update the job status to indicate it's ready for training
    const geometryServiceUrl = Deno.env.get('GEOMETRY_SERVICE_URL');
    
    if (geometryServiceUrl) {
      // Trigger training via geometry service
      const trainingResponse = await fetch(`${geometryServiceUrl}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_id: job.id,
          dataset_path: dataset.storage_path,
          model_version,
          epochs,
          batch_size,
          learning_rate,
          supabase_url: Deno.env.get('SUPABASE_URL'),
          supabase_key: Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')
        })
      });

      if (!trainingResponse.ok) {
        const error = await trainingResponse.text();
        throw new Error(`Training service error: ${error}`);
      }

      const trainingResult = await trainingResponse.json();
      console.log('Training started:', trainingResult);

      return new Response(
        JSON.stringify({ 
          success: true,
          job_id: job.id,
          status: 'training',
          message: 'Training started on geometry service'
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    } else {
      // No geometry service configured - manual training required
      await supabase
        .from('training_jobs')
        .update({
          status: 'pending',
          logs: [
            ...job.logs,
            {
              timestamp: new Date().toISOString(),
              message: 'Waiting for manual training trigger. GEOMETRY_SERVICE_URL not configured.',
              level: 'warning'
            }
          ]
        })
        .eq('id', job.id);

      return new Response(
        JSON.stringify({ 
          success: true,
          job_id: job.id,
          status: 'pending',
          message: 'Training job created. Manual training required - GEOMETRY_SERVICE_URL not configured.',
          instructions: 'Configure GEOMETRY_SERVICE_URL secret and implement training endpoint'
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

  } catch (error) {
    console.error('Error starting training:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return new Response(
      JSON.stringify({ error: errorMessage }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});