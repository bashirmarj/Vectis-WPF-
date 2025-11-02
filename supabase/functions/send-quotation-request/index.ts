import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { Resend } from "https://esm.sh/resend@4.0.0";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.3';

const resend = new Resend(Deno.env.get("RESEND_API_KEY"));

const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
const supabase = createClient(supabaseUrl, supabaseServiceKey);

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

interface FileInfo {
  name: string;
  content: string;
  size: number;
  quantity: number;
}

interface QuotationRequest {
  name: string;
  company?: string;
  email: string;
  phone: string;
  shippingAddress: string;
  message?: string;
  files: FileInfo[];
  drawingFiles?: FileInfo[];
}

interface StorageFileInfo {
  name: string;
  path: string;
  size: number;
  quantity: number;
}

// Helper function to hash IP address for privacy
async function hashIP(ip: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(ip);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

const handler = async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Extract IP address from request for logging (rate limiting disabled)
    const clientIP = req.headers.get('x-forwarded-for')?.split(',')[0].trim() || 
                     req.headers.get('x-real-ip') || 
                     'unknown';
    
    const ipHash = await hashIP(clientIP);

    const { 
      name, 
      company, 
      email, 
      phone, 
      shippingAddress,
      message,
      files,
      drawingFiles
    }: QuotationRequest = await req.json();

    console.log("Processing quotation request:", { 
      name, 
      company, 
      email, 
      phone, 
      filesCount: files.length,
      drawingFilesCount: drawingFiles?.length || 0
    });

    // Check total attachment size (Resend limit is 40MB)
    const totalSize = files.reduce((sum, f) => sum + f.size, 0) + 
                     (drawingFiles?.reduce((sum, f) => sum + f.size, 0) || 0);
    const totalSizeMB = totalSize / 1024 / 1024;
    
    console.log(`Total attachment size: ${totalSizeMB.toFixed(2)} MB`);
    
    if (totalSizeMB > 35) {
      console.error('Total attachment size exceeds limit:', totalSizeMB);
      return new Response(
        JSON.stringify({
          error: 'file_size_exceeded',
          message: 'Total file size exceeds 35MB limit',
          totalSizeMB
        }),
        {
          status: 400,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders,
          },
        }
      );
    }

    // Files are already base64 encoded from the client
    const attachments: Array<{ filename: string; content: string }> = [];

    // Add CAD files to attachments
    for (const file of files) {
      attachments.push({
        filename: file.name,
        content: file.content,
      });
    }

    // Add drawing files if provided
    if (drawingFiles && drawingFiles.length > 0) {
      for (const drawingFile of drawingFiles) {
        attachments.push({
          filename: drawingFile.name,
          content: drawingFile.content,
        });
      }
    }

    // Build file lists for email
    const cadFilesList = files.map((f, i) => 
      `<li>${f.name} - Quantity: ${f.quantity} (${(f.size / 1024 / 1024).toFixed(2)} MB)</li>`
    ).join('');

    const totalQuantity = files.reduce((sum, f) => sum + f.quantity, 0);

    const drawingFilesList = drawingFiles && drawingFiles.length > 0
      ? drawingFiles.map((f, i) => 
          `<li>${f.name} (${(f.size / 1024 / 1024).toFixed(2)} MB)</li>`
        ).join('')
      : null;

    // Record the submission with full customer details FIRST
    const { data: submission, error: insertError } = await supabase
      .from('quotation_submissions')
      .insert({
        email: email,
        customer_name: name,
        customer_company: company || null,
        customer_phone: phone,
        shipping_address: shippingAddress,
        customer_message: message || null,
        ip_hash: ipHash,
      })
      .select()
      .single();

    if (insertError || !submission) {
      console.error('Error recording submission:', insertError);
      throw new Error('Failed to create quotation record');
    }

    // Store file metadata in quote_line_items
    const lineItems = [
      ...files.map(file => ({
        quotation_id: submission.id,
        file_name: file.name,
        file_path: `${submission.id}/cad/${file.name}`,
        quantity: file.quantity,
      })),
      ...(drawingFiles || []).map(file => ({
        quotation_id: submission.id,
        file_name: file.name,
        file_path: `${submission.id}/drawings/${file.name}`,
        quantity: 1,
      }))
    ];

    let insertedLineItems: any[] = [];
    if (lineItems.length > 0) {
      const { data: inserted, error: lineItemsError } = await supabase
        .from('quote_line_items')
        .insert(lineItems)
        .select();

      if (lineItemsError) {
        console.error('Error storing line items:', lineItemsError);
      } else {
        insertedLineItems = inserted || [];
      }
    }

    // Trigger CAD analysis and preliminary pricing in background
    // Don't await - run asynchronously to avoid blocking response
    if (insertedLineItems.length > 0) {
      Promise.all(files.map(async (file, index) => {
        const lineItem = insertedLineItems[index];
        if (!lineItem) return;

        try {
          // Call analyze-cad function
          const analysisResponse = await fetch(`${supabaseUrl}/functions/v1/analyze-cad`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${supabaseServiceKey}`,
            },
            body: JSON.stringify({
              file_name: file.name,
              file_size: file.size,
              quantity: file.quantity
            })
          });

          if (!analysisResponse.ok) {
            console.error(`Analysis failed for ${file.name}`);
            return;
          }

          const analysisData = await analysisResponse.json();

          // Call pricing calculator
          const quoteResponse = await fetch(`${supabaseUrl}/functions/v1/calculate-preliminary-quote`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${supabaseServiceKey}`,
            },
            body: JSON.stringify({
              volume_cm3: analysisData.volume_cm3,
              surface_area_cm2: analysisData.surface_area_cm2,
              complexity_score: analysisData.complexity_score,
              quantity: file.quantity,
              process: 'CNC Machining',
              material: 'Aluminum 6061',
              finish: 'As-machined'
            })
          });

          if (!quoteResponse.ok) {
            console.error(`Quote calculation failed for ${file.name}`);
            return;
          }

          const quoteData = await quoteResponse.json();

          // Update line item with preliminary pricing and mesh_id
          const updateData: any = {
            estimated_volume_cm3: analysisData.volume_cm3,
            estimated_surface_area_cm2: analysisData.surface_area_cm2,
            estimated_complexity_score: analysisData.complexity_score,
            preliminary_unit_price: quoteData.unit_price,
            material_cost: quoteData.breakdown.material_cost,
            machining_cost: quoteData.breakdown.machining_cost,
            setup_cost: quoteData.breakdown.setup_cost,
            finish_cost: quoteData.breakdown.finish_cost,
            selected_process: quoteData.process,
            material_type: quoteData.material,
            finish_type: quoteData.finish,
            estimated_machine_time_hours: quoteData.estimated_hours
          };
          
          // Add mesh_id if available from analysis
          if (analysisData.mesh_id) {
            updateData.mesh_id = analysisData.mesh_id;
            console.log(`✅ Linking mesh_id ${analysisData.mesh_id} to line item ${lineItem.id}`);
          }
          
          await supabase
            .from('quote_line_items')
            .update(updateData)
            .eq('id', lineItem.id);

          console.log(`Preliminary quote generated for ${file.name}: $${quoteData.unit_price}`);

          // Save detected features using NEW manufacturing_features format
          if (analysisData.manufacturing_features || analysisData.feature_summary) {
            const features: any[] = [];
            const mfg = analysisData.manufacturing_features || {};
            
            // Process through-holes
            if (mfg.through_holes?.length > 0) {
              mfg.through_holes.forEach((hole: any, idx: number) => {
                features.push({
                  quotation_id: submission.id,
                  line_item_id: lineItem.id,
                  file_name: file.name,
                  feature_type: 'through_hole',
                  orientation: null,
                  parameters: { ...hole, index: idx }
                });
              });
            }
            
            // Process blind-holes
            if (mfg.blind_holes?.length > 0) {
              mfg.blind_holes.forEach((hole: any, idx: number) => {
                features.push({
                  quotation_id: submission.id,
                  line_item_id: lineItem.id,
                  file_name: file.name,
                  feature_type: 'blind_hole',
                  orientation: null,
                  parameters: { ...hole, index: idx }
                });
              });
            }
            
            // Process bores
            if (mfg.bores?.length > 0) {
              mfg.bores.forEach((bore: any, idx: number) => {
                features.push({
                  quotation_id: submission.id,
                  line_item_id: lineItem.id,
                  file_name: file.name,
                  feature_type: 'bore',
                  orientation: null,
                  parameters: { ...bore, index: idx }
                });
              });
            }
            
            // Process bosses
            if (mfg.bosses?.length > 0) {
              mfg.bosses.forEach((boss: any, idx: number) => {
                features.push({
                  quotation_id: submission.id,
                  line_item_id: lineItem.id,
                  file_name: file.name,
                  feature_type: 'boss',
                  orientation: null,
                  parameters: { ...boss, index: idx }
                });
              });
            }
            
            // Process planar faces
            if (mfg.planar_faces?.length > 0) {
              mfg.planar_faces.forEach((face: any, idx: number) => {
                features.push({
                  quotation_id: submission.id,
                  line_item_id: lineItem.id,
                  file_name: file.name,
                  feature_type: 'planar_face',
                  orientation: null,
                  parameters: { ...face, index: idx }
                });
              });
            }
            
            // Process fillets
            if (mfg.fillets?.length > 0) {
              mfg.fillets.forEach((fillet: any, idx: number) => {
                features.push({
                  quotation_id: submission.id,
                  line_item_id: lineItem.id,
                  file_name: file.name,
                  feature_type: 'fillet',
                  orientation: null,
                  parameters: { ...fillet, index: idx }
                });
              });
            }
            
            // Also save the summary for quick access
            if (analysisData.feature_summary) {
              features.push({
                quotation_id: submission.id,
                line_item_id: lineItem.id,
                file_name: file.name,
                feature_type: 'summary',
                orientation: null,
                parameters: analysisData.feature_summary
              });
            }
            
            if (features.length > 0) {
              const { error: featuresError } = await supabase
                .from('part_features')
                .insert(features);
              
              if (featuresError) {
                console.error(`Error saving features for ${file.name}:`, featuresError);
              } else {
                console.log(`Saved ${features.length} features for ${file.name}`);
              }
            }
          }
        } catch (error) {
          console.error(`Error processing ${file.name}:`, error);
        }
      })).catch(err => console.error('Background analysis error:', err));
    }

    // Send emails in the background to not block the response
    const sendEmails = async () => {
      try {
        // Read logo file
        const logoPath = '/var/task/public/logo-email.png';
        let logoContent: Uint8Array;
        
        try {
          logoContent = await Deno.readFile(logoPath);
        } catch (error) {
          console.error('Could not read logo file:', error);
          logoContent = new Uint8Array(0);
        }

        const [emailResponse, customerEmailResponse] = await Promise.all([
          // Admin email with attachments
          resend.emails.send({
            from: "Vectis Manufacturing <belmarj@vectismanufacturing.com>",
            to: ["belmarj@vectismanufacturing.com"],
            subject: `New Part Quotation Request - ${submission.quote_number}`,
            html: `
              <!DOCTYPE html>
              <html>
                <head>
                  <meta charset="utf-8">
                  <meta name="viewport" content="width=device-width, initial-scale=1.0">
                  <style>
                    body { margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #ffffff; }
                    .container { max-width: 600px; margin: 0 auto; padding: 40px 20px; }
                    .logo { text-align: center; margin-bottom: 20px; }
                    .logo img { height: 60px; width: auto; }
                    .header { margin-bottom: 30px; }
                    .company-name { font-size: 18px; font-weight: bold; color: #000000; margin-bottom: 20px; }
                    .intro { font-size: 14px; line-height: 1.6; color: #000000; margin-bottom: 30px; }
                    .info-table { width: 100%; border-collapse: collapse; margin-bottom: 30px; }
                    .info-table td { padding: 12px; border: 1px solid #cccccc; font-size: 14px; }
                    .info-table td:first-child { background-color: #f9f9f9; font-weight: normal; color: #000000; width: 180px; vertical-align: top; }
                    .info-table td:last-child { color: #000000; }
                    .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #cccccc; font-size: 12px; color: #666666; }
                    .file-list { margin: 0; padding: 0; list-style: none; }
                    .file-list li { margin: 4px 0; }
                  </style>
                </head>
                <body>
                  <div class="container">
                    <div class="logo">
                      <img src="cid:logo" alt="Vectis Manufacturing" />
                    </div>
                    <div class="header">
                      <div class="company-name">VECTIS MANUFACTURING</div>
                    </div>
                    
                    <div class="intro">
                      Hello, You have received a new Request for Quote. Your request will be handled in the order it was submitted. It could take up to 48 hours to receive a response, especially if your request is complex. Thank you for choosing Vectis Manufacturing! Here is a summary of the submission:
                    </div>
                    
                    <table class="info-table">
                      <tr>
                        <td>Quote Number</td>
                        <td>${submission.quote_number}</td>
                      </tr>
                      <tr>
                        <td>Date</td>
                        <td>${new Date().toLocaleDateString()}</td>
                      </tr>
                      <tr>
                        <td>Name</td>
                        <td>${name}</td>
                      </tr>
                      <tr>
                        <td>Company</td>
                        <td>${company || 'N/A'}</td>
                      </tr>
                      <tr>
                        <td>Email</td>
                        <td>${email}</td>
                      </tr>
                      <tr>
                        <td>Phone</td>
                        <td>${phone || 'N/A'}</td>
                      </tr>
                      <tr>
                        <td>Shipping Address</td>
                        <td>${shippingAddress}</td>
                      </tr>
                      <tr>
                        <td>Part Files</td>
                        <td>
                          <ul class="file-list">
                            ${files.map(f => `<li>${f.name} (${f.quantity} units)</li>`).join('')}
                          </ul>
                        </td>
                      </tr>
                      ${drawingFiles && drawingFiles.length > 0 ? `<tr>
                        <td>Technical Drawings</td>
                        <td>
                          <ul class="file-list">
                            ${drawingFiles.map((f: FileInfo) => `<li>${f.name}</li>`).join('')}
                          </ul>
                        </td>
                      </tr>` : ''}
                      ${message ? `<tr>
                        <td>Additional Information</td>
                        <td>${message}</td>
                      </tr>` : ''}
                    </table>
                    
                    <div class="footer">
                      ${attachments.length} file(s) attached to this email.
                    </div>
                  </div>
                </body>
              </html>
            `,
            attachments: attachments,
          }),
          // Customer confirmation email (without attachments)
          resend.emails.send({
            from: "Vectis Manufacturing <belmarj@vectismanufacturing.com>",
            to: [email],
            subject: `Quotation Request Received - ${submission.quote_number}`,
            html: `
              <!DOCTYPE html>
              <html>
                <head>
                  <meta charset="utf-8">
                  <meta name="viewport" content="width=device-width, initial-scale=1.0">
                </head>
                <body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #ffffff;">
                  <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
            <div style="text-align: center; margin-bottom: 20px;">
              <img src="https://inqabwlmvrvqsdrgskju.supabase.co/storage/v1/object/public/logos/logo-email.png" alt="Vectis Manufacturing" style="height: 60px; width: auto;" />
                    </div>
                    <div style="margin-bottom: 30px;">
                      <div style="font-size: 18px; font-weight: bold; color: #000000; margin-bottom: 20px;">VECTIS MANUFACTURING</div>
                    </div>

                    <div style="font-size: 14px; line-height: 1.6; color: #000000; margin-bottom: 30px;">
                      Hello ${name}, Thank you for your quotation request. We have successfully received your submission and our team will review it shortly. We will provide you with a detailed quote within 24-48 hours.
                    </div>

                    <!-- Quote Summary -->
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
                      <tr>
                        <td style="padding: 12px; border: 1px solid #cccccc; background-color: #f9f9f9; font-size: 14px; width: 180px;">Quote Number</td>
                        <td style="padding: 12px; border: 1px solid #cccccc; font-size: 14px; color: #000000;">${submission.quote_number}</td>
                      </tr>
                      <tr>
                        <td style="padding: 12px; border: 1px solid #cccccc; background-color: #f9f9f9; font-size: 14px;">Date</td>
                        <td style="padding: 12px; border: 1px solid #cccccc; font-size: 14px; color: #000000;">${new Date().toLocaleDateString()}</td>
                      </tr>
                      ${company ? `
                      <tr>
                        <td style="padding: 12px; border: 1px solid #cccccc; background-color: #f9f9f9; font-size: 14px;">Company</td>
                        <td style="padding: 12px; border: 1px solid #cccccc; font-size: 14px; color: #000000;">${company}</td>
                      </tr>` : ''}
                      <tr>
                        <td style="padding: 12px; border: 1px solid #cccccc; background-color: #f9f9f9; font-size: 14px;">Phone</td>
                        <td style="padding: 12px; border: 1px solid #cccccc; font-size: 14px; color: #000000;">${phone}</td>
                      </tr>
                      <tr>
                        <td style="padding: 12px; border: 1px solid #cccccc; background-color: #f9f9f9; font-size: 14px;">Total Parts</td>
                        <td style="padding: 12px; border: 1px solid #cccccc; font-size: 14px; color: #000000;">${totalQuantity}</td>
                      </tr>
                      <tr>
                        <td style="padding: 12px; border: 1px solid #cccccc; background-color: #f9f9f9; font-size: 14px; vertical-align: top;">Shipping Address</td>
                        <td style="padding: 12px; border: 1px solid #cccccc; font-size: 14px; color: #000000; white-space: pre-line;">${shippingAddress}</td>
                      </tr>
                      ${message ? `
                      <tr>
                        <td style="padding: 12px; border: 1px solid #cccccc; background-color: #f9f9f9; font-size: 14px; vertical-align: top;">Additional Information</td>
                        <td style="padding: 12px; border: 1px solid #cccccc; font-size: 14px; color: #000000; white-space: pre-line;">${message}</td>
                      </tr>` : ''}
                    </table>

                    <!-- Submitted Files -->
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
                      <tr>
                        <td style="padding: 12px; border: 1px solid #cccccc; background-color: #f9f9f9; font-size: 14px; vertical-align: top; width: 180px;">CAD Files (${files.length})</td>
                        <td style="padding: 12px; border: 1px solid #cccccc; font-size: 14px; color: #000000;">
                          <ul style="margin: 0; padding: 0; list-style: none;">
                            ${files.map(f => `<li style="margin: 4px 0;">${f.name} - Quantity: ${f.quantity}</li>`).join('')}
                          </ul>
                        </td>
                      </tr>
                      ${drawingFiles && drawingFiles.length > 0 ? `
                      <tr>
                        <td style="padding: 12px; border: 1px solid #cccccc; background-color: #f9f9f9; font-size: 14px; vertical-align: top;">Technical Drawings (${drawingFiles.length})</td>
                        <td style="padding: 12px; border: 1px solid #cccccc; font-size: 14px; color: #000000;">
                          <ul style="margin: 0; padding: 0; list-style: none;">
                            ${drawingFiles.map(f => `<li style="margin: 4px 0;">${f.name}</li>`).join('')}
                          </ul>
                        </td>
                      </tr>` : ''}
                    </table>

                    <div style="font-size: 14px; line-height: 1.6; color: #000000; margin-bottom: 30px;">
                      If you have any questions or need to make changes to your request, please reply to this email and reference your quote number.
                    </div>

                    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #cccccc; font-size: 12px; color: #666666;">
                      Thank you for choosing Vectis Manufacturing.
                    </div>
                  </div>
                </body>
              </html>
            `
          })
        ]);

        console.log("Admin email sent:", emailResponse);
        console.log("Customer email sent:", customerEmailResponse);
      } catch (emailError) {
        console.error("Error sending emails in background:", emailError);
      }
    };

    // Start background email task (non-blocking)
    (globalThis as any).EdgeRuntime?.waitUntil(sendEmails());

    // Return immediate response without waiting for emails

    return new Response(JSON.stringify({ 
      success: true, 
      quoteNumber: submission.quote_number
    }), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        ...corsHeaders,
      },
    });
  } catch (error: any) {
    console.error("Error in send-quotation-request function:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    );
  }
};

serve(handler);
