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

    const { dataset_name } = await req.json();

    console.log(`Setting up dataset: ${dataset_name}`);

    // Update dataset status to downloading
    const { error: updateError } = await supabase
      .from('training_datasets')
      .update({ 
        download_status: 'downloading',
        download_progress: 0 
      })
      .eq('name', dataset_name);

    if (updateError) {
      throw new Error(`Failed to update dataset status: ${updateError.message}`);
    }

    // For MFCAD dataset, we'll create the initial directory structure
    // Note: The actual dataset files need to be uploaded manually or via a separate download process
    // as they're typically large (several GB) and hosted on external platforms
    
    const directories = [
      `${dataset_name}/graph/`,
      `${dataset_name}/labels/`,
      `${dataset_name}/split/`
    ];

    // Create directory marker files in storage
    for (const dir of directories) {
      await supabase.storage
        .from('training-datasets')
        .upload(`${dir}.keep`, new Blob([''], { type: 'text/plain' }), {
          upsert: true
        });
    }

    // Create a README file with instructions
    const readme = `# ${dataset_name} Dataset Setup

This dataset requires manual download from the source repository.

## Download Instructions:

1. Visit: https://github.com/hducg/UV-Net
2. Follow their instructions to download the MFCAD dataset
3. The dataset should contain:
   - graph/*.bin files (DGL binary format)
   - labels/*_ids.json files
   - split.json (train/val/test split)

## Upload Instructions:

Once downloaded, upload files to this storage bucket:
- Upload all .bin files to: ${dataset_name}/graph/
- Upload all _ids.json files to: ${dataset_name}/labels/
- Upload split.json to: ${dataset_name}/split/

## File Format:

- Graph files: DGL binary format containing B-Rep face adjacency graphs
- Label files: JSON format with feature class labels per face
- Split file: JSON with train/validation/test file lists

## Classes (16 total):

0. rectangular_through_slot
1. triangular_through_slot
2. rectangular_passage
3. triangular_passage
4. 6sides_passage
5. rectangular_through_step
6. 2sides_through_step
7. slanted_through_step
8. rectangular_blind_step
9. triangular_blind_step
10. rectangular_blind_slot
11. rectangular_pocket
12. triangular_pocket
13. 6sides_pocket
14. chamfer
15. stock

After uploading, mark the dataset as 'completed' in the training_datasets table.
`;

    await supabase.storage
      .from('training-datasets')
      .upload(`${dataset_name}/README.md`, new Blob([readme], { type: 'text/markdown' }), {
        upsert: true
      });

    // Update dataset with instructions
    const { error: finalUpdateError } = await supabase
      .from('training_datasets')
      .update({ 
        download_status: 'pending',
        storage_path: dataset_name,
        metadata: {
          instructions: 'Manual download required. See README.md in storage bucket.',
          setup_completed: new Date().toISOString()
        }
      })
      .eq('name', dataset_name);

    if (finalUpdateError) {
      throw new Error(`Failed to update dataset: ${finalUpdateError.message}`);
    }

    return new Response(
      JSON.stringify({ 
        success: true,
        message: 'Dataset structure created. Manual download required.',
        storage_path: dataset_name,
        instructions: 'Check README.md in the training-datasets bucket for download and upload instructions.'
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Error setting up dataset:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return new Response(
      JSON.stringify({ error: errorMessage }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});