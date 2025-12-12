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

interface QuoteEmailRequest {
  quotationId: string;
  customerEmail: string;
  customerName: string;
  quoteNumber: string;
}

// Unified email template generator
function generateUnifiedEmailTemplate(options: {
  heroTitle: string;
  heroSubtitle: string;
  quoteNumber: string;
  statusStep: 1 | 2 | 3;
  detailsContent: string;
  lineItemsContent?: string;
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
    lineItemsContent,
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
        ${timelineText}
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
                  <div style="display: inline-block; width: 64px; height: 64px; border-radius: 50%; background-color: #d1fae5; margin-bottom: 20px; line-height: 64px;">
                    <span style="font-size: 32px; color: #10b981; line-height: 64px; font-family: Arial, sans-serif;">&#10003;</span>
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
                      <h3 style="color: #1e40af; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin: 0;">Quote Details</h3>
                    </div>
                    
                    ${detailsContent}
                  </div>

                  ${lineItemsContent || ''}

                  ${timelineBox}

                  <p style="text-align: center; color: #64748b; font-size: 14px; margin-top: 30px;">
                    Questions about this quote? <a href="mailto:belmarj@vectismanufacturing.com" style="color: #3b82f6; text-decoration: none; font-weight: 600;">Reply to this email</a>
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

// Helper to generate line items table
function generateLineItemsTable(lineItems: any[]): string {
  const rows = lineItems.map(item => `
    <tr>
      <td style="padding: 12px 15px; border-bottom: 1px solid #e2e8f0; font-size: 13px; color: #334155;">${item.file_name}</td>
      <td style="padding: 12px 15px; border-bottom: 1px solid #e2e8f0; font-size: 13px; color: #334155; text-align: center;">${item.quantity}</td>
      <td style="padding: 12px 15px; border-bottom: 1px solid #e2e8f0; font-size: 13px; color: #334155; text-align: right;">$${Number(item.unit_price || 0).toFixed(2)}</td>
      <td style="padding: 12px 15px; border-bottom: 1px solid #e2e8f0; font-size: 13px; color: #334155; text-align: right; font-weight: 600;">$${(Number(item.unit_price || 0) * item.quantity).toFixed(2)}</td>
    </tr>
    ${item.notes ? `
    <tr>
      <td colspan="4" style="padding: 8px 15px; border-bottom: 1px solid #e2e8f0; font-size: 12px; color: #64748b; font-style: italic; background-color: rgba(248, 250, 252, 0.5);">
        ${item.notes}
      </td>
    </tr>
    ` : ''}
  `).join('');

  return `
    <!-- Line Items Table -->
    <div style="background-color: rgba(248, 250, 252, 0.85); border: 1px solid #e2e8f0; border-radius: 6px; padding: 0; margin-top: 20px; overflow: hidden;">
      <div style="background-color: rgba(239, 246, 255, 0.9); padding: 12px 20px; border-bottom: 1px solid #dbeafe;">
        <h3 style="color: #1e40af; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin: 0;">Line Items</h3>
      </div>
      <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%">
        <thead>
          <tr style="background-color: rgba(241, 245, 249, 0.9);">
            <th style="padding: 12px 15px; text-align: left; font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Part</th>
            <th style="padding: 12px 15px; text-align: center; font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Qty</th>
            <th style="padding: 12px 15px; text-align: right; font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Unit Price</th>
            <th style="padding: 12px 15px; text-align: right; font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Total</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
    </div>
  `;
}

// Helper to generate totals section
function generateTotalsSection(quote: any): string {
  return `
    <!-- Totals Section -->
    <div style="background-color: rgba(248, 250, 252, 0.85); border: 1px solid #e2e8f0; border-radius: 6px; padding: 0; margin-top: 20px; overflow: hidden;">
      <div style="padding: 12px 20px; border-bottom: 1px solid #e2e8f0;">
        <span style="color: #64748b; font-size: 13px; font-weight: 600; float: left;">Subtotal</span>
        <span style="color: #1e293b; font-size: 13px; font-weight: 600; float: right;">$${Number(quote.subtotal).toFixed(2)}</span>
        <div style="clear: both;"></div>
      </div>
      <div style="padding: 12px 20px; border-bottom: 1px solid #e2e8f0;">
        <span style="color: #64748b; font-size: 13px; font-weight: 600; float: left;">Shipping</span>
        <span style="color: #1e293b; font-size: 13px; font-weight: 600; float: right;">$${Number(quote.shipping_cost).toFixed(2)}</span>
        <div style="clear: both;"></div>
      </div>
      <div style="padding: 12px 20px; border-bottom: 1px solid #e2e8f0;">
        <span style="color: #64748b; font-size: 13px; font-weight: 600; float: left;">Tax (${Number(quote.tax_rate).toFixed(2)}%)</span>
        <span style="color: #1e293b; font-size: 13px; font-weight: 600; float: right;">$${Number(quote.tax_amount).toFixed(2)}</span>
        <div style="clear: both;"></div>
      </div>
      <div style="padding: 15px 20px; background-color: rgba(16, 185, 129, 0.1);">
        <span style="color: #1e293b; font-size: 15px; font-weight: 700; float: left;">Total</span>
        <span style="color: #10b981; font-size: 18px; font-weight: 700; float: right;">$${Number(quote.total_amount).toFixed(2)} ${quote.currency}</span>
        <div style="clear: both;"></div>
      </div>
    </div>
  `;
}

const handler = async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { quotationId, customerEmail, customerName, quoteNumber }: QuoteEmailRequest = await req.json();

    console.log('Fetching quote data for:', quoteNumber);

    // Fetch quote details
    const { data: quote, error: quoteError } = await supabase
      .from('quotes')
      .select('*')
      .eq('quotation_id', quotationId)
      .single();

    if (quoteError || !quote) {
      console.error('Error fetching quote:', quoteError);
      throw new Error('Quote not found');
    }

    // Fetch line items
    const { data: lineItems, error: lineItemsError } = await supabase
      .from('quote_line_items')
      .select('*')
      .eq('quotation_id', quotationId)
      .order('created_at', { ascending: true });

    if (lineItemsError) {
      console.error('Error fetching line items:', lineItemsError);
      throw lineItemsError;
    }

    // Generate quote details content
    const detailsContent = `
      ${generateDetailRow('Quote Number', quoteNumber)}
      ${generateDetailRow('Date', new Date().toLocaleDateString())}
      ${quote.estimated_lead_time_days ? generateDetailRow('Estimated Lead Time', `${quote.estimated_lead_time_days} business days`) : ''}
      ${generateDetailRow('Valid Until', new Date(quote.valid_until).toLocaleDateString())}
    `;

    // Build line items and totals content
    const lineItemsContent = `
      ${generateLineItemsTable(lineItems || [])}
      ${generateTotalsSection(quote)}
      ${quote.notes ? `
        <div style="background-color: rgba(248, 250, 252, 0.85); border: 1px solid #e2e8f0; border-radius: 6px; padding: 15px 20px; margin-top: 20px;">
          <span style="color: #64748b; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; display: block; margin-bottom: 8px;">Additional Notes</span>
          <p style="color: #334155; font-size: 14px; margin: 0; line-height: 1.6; white-space: pre-line;">${quote.notes}</p>
        </div>
      ` : ''}
    `;

    // Calculate days until expiry
    const validUntil = new Date(quote.valid_until);
    const today = new Date();
    const daysUntilExpiry = Math.ceil((validUntil.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
    const expiryText = daysUntilExpiry > 0 
      ? `&#128197; This quote is valid for <strong>${daysUntilExpiry} days</strong>`
      : `&#9888; This quote has expired`;

    const emailHtml = generateUnifiedEmailTemplate({
      heroTitle: 'Your Quote is Ready',
      heroSubtitle: `Hello <strong>${customerName}</strong>,<br>We've completed your custom manufacturing quote.`,
      quoteNumber: quoteNumber,
      statusStep: 3,
      detailsContent: detailsContent,
      lineItemsContent: lineItemsContent,
      timelineText: expiryText,
      footerText: 'Thank you for choosing Vectis Manufacturing.'
    });

    console.log('Sending email to:', customerEmail);

    const emailResponse = await resend.emails.send({
      from: "Vectis Manufacturing <belmarj@vectismanufacturing.com>",
      to: [customerEmail],
      subject: `Your Quote ${quoteNumber} is Ready`,
      html: emailHtml
    });

    console.log("Email sent successfully:", emailResponse);

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        ...corsHeaders,
      },
    });
  } catch (error: any) {
    console.error("Error sending quote email:", error);
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
