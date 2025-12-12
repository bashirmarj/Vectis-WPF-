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
  geometric_features?: any;
  mesh_data?: {
    vertices: number[];
    indices: number[];
    normals: number[];
    triangle_count: number;
    vertex_face_ids?: number[];
  };
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

// Helper function to hash strings
async function hashString(input: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// Unified email template generator
function generateUnifiedEmailTemplate(options: {
  heroTitle: string;
  heroSubtitle: string;
  quoteNumber: string;
  statusStep: 1 | 2 | 3; // 1=Received, 2=Reviewing, 3=Quote Ready
  detailsContent: string;
  fileListContent?: string;
  timelineText?: string;
  showStatusTracker?: boolean;
  footerText?: string;
}): string {
  const {
    heroTitle,
    heroSubtitle,
    quoteNumber,
    statusStep,
    detailsContent,
    fileListContent,
    timelineText,
    showStatusTracker = true,
    footerText
  } = options;

  const statusTracker = showStatusTracker ? `
    <!-- 2. Visual Status Tracker -->
    <div style="background-color: rgba(248, 250, 252, 0.9); padding: 25px 20px; border-bottom: 1px solid #e2e8f0; text-align: center;">
      <div style="display: inline-block; width: 30%; vertical-align: top; position: relative;">
        <span style="height: 12px; width: 12px; background-color: ${statusStep >= 1 ? '#10b981' : '#cbd5e1'}; border-radius: 50%; display: inline-block; margin-bottom: 8px; ${statusStep >= 1 ? 'box-shadow: 0 0 0 4px #d1fae5;' : ''}"></span>
        <span style="font-size: 11px; color: ${statusStep >= 1 ? '#10b981' : '#64748b'}; font-weight: 600; text-transform: uppercase; display: block; letter-spacing: 0.5px;">Received</span>
      </div><!--
      --><div style="display: inline-block; width: 30%; vertical-align: top; position: relative;">
        <span style="height: 12px; width: 12px; background-color: ${statusStep >= 2 ? '#10b981' : '#cbd5e1'}; border-radius: 50%; display: inline-block; margin-bottom: 8px; ${statusStep >= 2 ? 'box-shadow: 0 0 0 4px #d1fae5;' : ''}"></span>
        <span style="font-size: 11px; color: ${statusStep >= 2 ? '#10b981' : '#64748b'}; font-weight: 600; text-transform: uppercase; display: block; letter-spacing: 0.5px;">Reviewing</span>
      </div><!--
      --><div style="display: inline-block; width: 30%; vertical-align: top; position: relative;">
        <span style="height: 12px; width: 12px; background-color: ${statusStep >= 3 ? '#10b981' : '#cbd5e1'}; border-radius: 50%; display: inline-block; margin-bottom: 8px; ${statusStep >= 3 ? 'box-shadow: 0 0 0 4px #d1fae5;' : ''}"></span>
        <span style="font-size: 11px; color: ${statusStep >= 3 ? '#10b981' : '#64748b'}; font-weight: 600; text-transform: uppercase; display: block; letter-spacing: 0.5px;">Quote Ready</span>
      </div>
    </div>
  ` : '';

  const timelineBox = timelineText ? `
    <!-- Timeline / Next Steps -->
    <div style="margin-top: 30px; text-align: center; padding: 20px; background-color: rgba(255, 251, 235, 0.9); border: 1px solid #fcd34d; border-radius: 6px;">
      <p style="color: #92400e; font-size: 14px; font-weight: 500; margin: 0;">
        &#9201; ${timelineText}
      </p>
    </div>
  ` : '';

  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>${heroTitle}</title>
      <!--[if mso]>
      <noscript>
        <xml>
          <o:OfficeDocumentSettings>
            <o:PixelsPerInch>96</o:PixelsPerInch>
          </o:OfficeDocumentSettings>
        </xml>
      </noscript>
      <![endif]-->
    </head>
    <body style="margin: 0; padding: 0; background-color: #f1f4f9; font-family: 'Segoe UI', Helvetica, Arial, sans-serif; -webkit-font-smoothing: antialiased;">

    <div style="width: 100%; table-layout: fixed; background-color: #f1f4f9; padding-bottom: 40px;">
      <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%">
        <tr>
          <td align="center">
            
            <!-- Spacer -->
            <div style="height: 40px;"></div>

            <div style="margin: 0 auto; max-width: 600px; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); background-image: url('https://res.cloudinary.com/dbcfeio6b/image/upload/v1765512246/LOGO_edi8ss.png'); background-repeat: no-repeat; background-position: center 120px; background-size: 80%;">
              <div style="background-color: rgba(255, 255, 255, 0.93); width: 100%; height: 100%;">
              
                <!-- 1. Brand Header -->
                <div style="background-color: #000000; padding: 30px 40px; text-align: center; position: relative; z-index: 2;">
                  <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%">
                    <tr>
                      <td align="center" style="vertical-align: middle;">
                        <img src="https://res.cloudinary.com/dbcfeio6b/image/upload/v1765508292/output-onlinepngtools-removebg-preview_1_kkhayz.png" alt="VM Logo" width="88" style="display: inline-block; vertical-align: middle; margin-right: 15px; height: auto; border: 0;">
                        <span style="color: #ffffff; font-size: 24px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; display: inline-block; vertical-align: middle;">Vectis Manufacturing</span>
                      </td>
                    </tr>
                  </table>
                </div>

                ${statusTracker}

                <!-- 3. Hero Section -->
                <div style="padding: 40px 40px 20px 40px; text-align: center;">
                  <div style="display: inline-block; width: 64px; height: 64px; border-radius: 50%; background-color: #e0f2fe; margin-bottom: 20px; line-height: 64px;">
                    <span style="font-size: 32px; color: #0284c7; line-height: 64px; font-family: Arial, sans-serif;">&#10003;</span>
                  </div>
                  
                  <h2 style="color: #1e293b; font-size: 22px; font-weight: 700; margin: 0 0 10px 0;">${heroTitle}</h2>
                  <p style="color: #64748b; font-size: 16px; margin: 0; line-height: 1.5;">${heroSubtitle}</p>
                </div>

                <!-- 4. Content & Details -->
                <div style="padding: 0 40px 40px 40px;">
                  
                  <!-- Reference Number Block -->
                  <div style="text-align: center; margin-bottom: 25px;">
                    <span style="background: #e2e8f0; color: #475569; padding: 6px 12px; border-radius: 15px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px;">REF: ${quoteNumber}</span>
                  </div>

                  <!-- Details "Receipt" Card -->
                  <div style="background-color: rgba(248, 250, 252, 0.85); border: 1px solid #e2e8f0; border-radius: 6px; padding: 0; margin-top: 25px; overflow: hidden;">
                    <div style="background-color: rgba(239, 246, 255, 0.9); padding: 12px 20px; border-bottom: 1px solid #dbeafe;">
                      <h3 style="color: #1e40af; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin: 0;">Order Summary</h3>
                    </div>
                    
                    ${detailsContent}
                    
                    ${fileListContent || ''}
                  </div>

                  ${timelineBox}

                  <p style="text-align: center; color: #64748b; font-size: 14px; margin-top: 30px;">
                    Need to make changes? <a href="mailto:belmarj@vectismanufacturing.com" style="color: #3b82f6; text-decoration: none; font-weight: 600;">Reply to this email</a>
                  </p>

                </div>

              </div>
            </div>

            <!-- Footer -->
            <div style="background-color: #f1f4f9; padding: 30px; text-align: center; font-size: 12px; color: #94a3b8;">
              <p style="margin-bottom: 10px;">&copy; ${new Date().getFullYear()} Vectis Manufacturing. All rights reserved.</p>
              ${footerText ? `<p>${footerText}</p>` : ''}
            </div>

            <!-- Spacer -->
            <div style="height: 40px;"></div>

          </td>
        </tr>
      </table>
    </div>

    </body>
    </html>
  `;
}

// Helper to generate detail rows
function generateDetailRow(label: string, value: string): string {
  return `
    <div style="padding: 12px 20px; border-bottom: 1px solid #e2e8f0;">
      <span style="color: #64748b; font-size: 13px; font-weight: 600; float: left; width: 40%;">${label}</span>
      <span style="color: #1e293b; font-size: 13px; font-weight: 600; float: right; width: 60%; text-align: right;">${value}</span>
      <div style="clear: both;"></div>
    </div>
  `;
}

// Helper to generate file list
function generateFileList(files: FileInfo[], drawingFiles?: FileInfo[]): string {
  const fileItems = files.map(f => `
    <div style="background-color: rgba(255, 255, 255, 0.8); border: 1px solid #e2e8f0; border-radius: 4px; padding: 10px; margin-top: 10px;">
      <span style="display: inline-block; width: 12px; height: 16px; border: 2px solid #64748b; border-radius: 2px; vertical-align: middle; margin-right: 8px; position: relative; top: -1px;"></span>
      <span style="font-size: 14px; color: #334155; font-weight: 500; vertical-align: middle;">${f.name}</span>
      <span style="float: right; font-size: 12px; color: #64748b; font-weight: 600;">x${f.quantity}</span>
    </div>
  `).join('');

  const drawingItems = drawingFiles && drawingFiles.length > 0 ? drawingFiles.map(f => `
    <div style="background-color: rgba(255, 255, 255, 0.8); border: 1px solid #e2e8f0; border-radius: 4px; padding: 10px; margin-top: 10px;">
      <span style="display: inline-block; width: 12px; height: 16px; border: 2px solid #64748b; border-radius: 2px; vertical-align: middle; margin-right: 8px; position: relative; top: -1px;"></span>
      <span style="font-size: 14px; color: #334155; font-weight: 500; vertical-align: middle;">${f.name}</span>
      <span style="float: right; font-size: 12px; color: #64748b; font-weight: 600;">Drawing</span>
    </div>
  `).join('') : '';

  return `
    <div style="padding: 12px 20px; background-color: rgba(255,255,255,0.7);">
      <span style="color: #64748b; font-size: 13px; font-weight: 600; width: 100%; margin-bottom: 8px; display: block;">Uploaded Files</span>
      ${fileItems}
      ${drawingItems}
    </div>
  `;
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

    const totalQuantity = files.reduce((sum, f) => sum + f.quantity, 0);

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
          // Save mesh data and geometric_features to cad_meshes table
          if (file.mesh_data) {
            const fileHash = await hashString(file.name + file.size);
            
            const { data: meshData, error: meshError } = await supabase
              .from('cad_meshes')
              .insert({
                quotation_id: submission.id,
                line_item_id: lineItem.id,
                file_name: file.name,
                file_hash: fileHash,
                vertices: file.mesh_data.vertices || [],
                indices: file.mesh_data.indices || [],
                normals: file.mesh_data.normals || [],
                triangle_count: file.mesh_data.triangle_count || 0,
                vertex_face_ids: file.mesh_data.vertex_face_ids || [],
                geometric_features: file.geometric_features || null
              })
              .select('id')
              .single();
            
            if (!meshError && meshData) {
              // Update line item with mesh_id
              await supabase
                .from('quote_line_items')
                .update({ mesh_id: meshData.id })
                .eq('id', lineItem.id);
              
              console.log(`✅ Saved mesh data for ${file.name} with geometric_features`);
            } else if (meshError) {
              console.error(`Error saving mesh for ${file.name}:`, meshError);
            }
          }
          
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

          // Update line item with preliminary pricing
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
        // Format shipping address for display
        const formattedAddress = shippingAddress.split('\n').join(', ');

        // Generate admin email content
        const adminDetailsContent = `
          ${generateDetailRow('Quote Number', submission.quote_number)}
          ${generateDetailRow('Date', new Date().toLocaleDateString())}
          ${generateDetailRow('Name', name)}
          ${generateDetailRow('Company', company || 'N/A')}
          ${generateDetailRow('Email', email)}
          ${generateDetailRow('Phone', phone || 'N/A')}
          ${generateDetailRow('Shipping Address', formattedAddress)}
          ${message ? generateDetailRow('Additional Notes', message) : ''}
        `;

        const adminEmailHtml = generateUnifiedEmailTemplate({
          heroTitle: 'New Quote Request',
          heroSubtitle: `A new quotation request has been submitted by <strong>${name}</strong>.<br>Please review the details below.`,
          quoteNumber: submission.quote_number,
          statusStep: 1,
          detailsContent: adminDetailsContent,
          fileListContent: generateFileList(files, drawingFiles),
          timelineText: `Review and respond within <strong>24-48 Hours</strong>`,
          footerText: `${attachments.length} file(s) attached to this email.`
        });

        // Generate customer email content
        const customerDetailsContent = `
          ${generateDetailRow('Company', company || 'N/A')}
          ${generateDetailRow('Total Quantity', `${totalQuantity} Parts`)}
          ${generateDetailRow('Shipping To', formattedAddress)}
        `;

        const customerEmailHtml = generateUnifiedEmailTemplate({
          heroTitle: 'Request Received',
          heroSubtitle: `Thanks for your request, <strong>${name}</strong>.<br>Our engineering team has started reviewing your files.`,
          quoteNumber: submission.quote_number,
          statusStep: 1,
          detailsContent: customerDetailsContent,
          fileListContent: generateFileList(files, drawingFiles),
          timelineText: `Estimated Response Time: <strong>24-48 Hours</strong>`
        });

        const [emailResponse, customerEmailResponse] = await Promise.all([
          // Admin email with attachments
          resend.emails.send({
            from: "Vectis Manufacturing <belmarj@vectismanufacturing.com>",
            to: ["belmarj@vectismanufacturing.com"],
            subject: `New Part Quotation Request - ${submission.quote_number}`,
            html: adminEmailHtml,
            attachments: attachments,
          }),
          // Customer confirmation email (without attachments)
          resend.emails.send({
            from: "Vectis Manufacturing <belmarj@vectismanufacturing.com>",
            to: [email],
            subject: `Quotation Request Received - ${submission.quote_number}`,
            html: customerEmailHtml
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
