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

    const url = new URL(req.url);
    const job_id = url.searchParams.get('job_id');

    if (job_id) {
      // Get specific job
      const { data: job, error } = await supabase
        .from('training_jobs')
        .select('*')
        .eq('id', job_id)
        .single();

      if (error) {
        throw new Error(`Job not found: ${error.message}`);
      }

      return new Response(
        JSON.stringify(job),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    } else {
      // Get all jobs (latest first)
      const { data: jobs, error } = await supabase
        .from('training_jobs')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(50);

      if (error) {
        throw new Error(`Failed to fetch jobs: ${error.message}`);
      }

      // Also get datasets status
      const { data: datasets } = await supabase
        .from('training_datasets')
        .select('*')
        .order('created_at', { ascending: false });

      return new Response(
        JSON.stringify({ 
          jobs: jobs || [],
          datasets: datasets || []
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

  } catch (error) {
    console.error('Error fetching training status:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return new Response(
      JSON.stringify({ error: errorMessage }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});