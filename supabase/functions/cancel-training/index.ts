import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    console.log('Cancel training function called');

    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      {
        global: {
          headers: { Authorization: req.headers.get('Authorization')! },
        },
        auth: {
          persistSession: false,
        },
      }
    )

    // When verify_jwt = true, Supabase validates the JWT automatically
    // We just need to get the user from the validated token
    const { data: { user }, error: authError } = await supabaseClient.auth.getUser()
    
    if (authError || !user) {
      console.error('Auth error:', authError);
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    console.log('User authenticated:', user.id);

    // Check if user is admin using the has_role function
    const { data: hasAdminRole, error: roleError } = await supabaseClient
      .rpc('has_role', { _user_id: user.id, _role: 'admin' })

    if (roleError || !hasAdminRole) {
      console.error('Role check failed:', roleError, 'hasAdminRole:', hasAdminRole);
      return new Response(
        JSON.stringify({ error: 'Admin access required' }),
        { status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    console.log('Admin access verified');

    const { job_id } = await req.json()

    if (!job_id) {
      return new Response(
        JSON.stringify({ error: 'job_id is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    console.log('Attempting to cancel job:', job_id);

    // Get current job status
    const { data: job, error: fetchError } = await supabaseClient
      .from('training_jobs')
      .select('id, status, dataset_name')
      .eq('id', job_id)
      .single()

    if (fetchError || !job) {
      console.error('Job fetch error:', fetchError);
      return new Response(
        JSON.stringify({ error: 'Training job not found' }),
        { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    console.log('Current job status:', job.status);

    // Check if job can be cancelled
    const cancellableStatuses = ['pending', 'pending_local', 'running']
    if (!cancellableStatuses.includes(job.status)) {
      return new Response(
        JSON.stringify({ 
          error: `Cannot cancel job with status: ${job.status}. Only pending, pending_local, or running jobs can be cancelled.` 
        }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Update job status to cancelled
    const { error: updateError } = await supabaseClient
      .from('training_jobs')
      .update({ 
        status: 'cancelled',
        cancelled_at: new Date().toISOString(),
        cancelled_by: user.id,
        error_message: 'Training cancelled by user'
      })
      .eq('id', job_id)

    if (updateError) {
      console.error('Error cancelling training job:', updateError)
      return new Response(
        JSON.stringify({ error: 'Failed to cancel training job' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    console.log(`Training job ${job_id} (${job.dataset_name}) cancelled by user ${user.id}`)

    return new Response(
      JSON.stringify({ 
        success: true,
        message: `Training job "${job.dataset_name}" has been cancelled`,
        job_id 
      }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Error in cancel-training function:', error)
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
    return new Response(
      JSON.stringify({ error: errorMessage }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})

