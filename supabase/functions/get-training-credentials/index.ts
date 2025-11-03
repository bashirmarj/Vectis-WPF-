import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      {
        global: {
          headers: { Authorization: req.headers.get('Authorization')! },
        },
      }
    );

    // Verify user is authenticated
    const { data: { user }, error: authError } = await supabaseClient.auth.getUser();
    if (authError || !user) {
      console.error('Authentication failed:', authError);
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Check if user has admin role
    const { data: roleData, error: roleError } = await supabaseClient
      .from('user_roles')
      .select('role')
      .eq('user_id', user.id)
      .eq('role', 'admin')
      .single();

    if (roleError || !roleData) {
      console.error('User is not an admin:', user.id);
      return new Response(
        JSON.stringify({ error: 'Admin access required' }),
        { status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Log the credential request for security audit
    console.log(`Training credentials requested by admin user: ${user.id} (${user.email})`);

    // Return the credentials needed for local training
    const credentials = {
      supabase_url: Deno.env.get('SUPABASE_URL'),
      supabase_key: Deno.env.get('SUPABASE_SERVICE_ROLE_KEY'),
      instructions: {
        step1: 'Create a file named .env in the training/ directory',
        step2: 'Copy the following content into training/.env:',
        content: `SUPABASE_URL=${Deno.env.get('SUPABASE_URL')}
SUPABASE_KEY=${Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')}

# Optional: Adjust these training parameters
BATCH_SIZE=8
EPOCHS=100
LEARNING_RATE=0.001`,
        step3: 'Run: python training/check_gpu.py to verify your setup',
        step4: 'Start training using the command from the dashboard'
      },
      warning: '⚠️  IMPORTANT: Keep the SUPABASE_KEY secret! Never commit the .env file to git.'
    };

    return new Response(
      JSON.stringify(credentials),
      { 
        status: 200, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error('Error in get-training-credentials:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
