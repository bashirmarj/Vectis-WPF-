import { serve } from "https://deno.land/std@0.190.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface ConfirmationRequest {
  customerName: string;
  customerEmail: string;
  quoteNumber: string;
  company?: string;
  phone?: string;
  address?: string;
  projectDescription?: string;
  partDetails?: {
    partName?: string;
    material?: string;
    quantity?: number;
    finish?: string;
    heatTreatment?: boolean;
    heatTreatmentDetails?: string;
    threadsTolerances?: string;
  };
}

// Gmail OAuth2 functions
async function getAccessToken(): Promise<string> {
  const clientId = Deno.env.get("GMAIL_CLIENT_ID")!;
  const clientSecret = Deno.env.get("GMAIL_CLIENT_SECRET")!;
  const refreshToken = Deno.env.get("GMAIL_REFRESH_TOKEN")!;

  const response = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id: clientId,
      client_secret: clientSecret,
      refresh_token: refreshToken,
      grant_type: "refresh_token",
    }),
  });

  const data = await response.json();
  if (!data.access_token) {
    console.error("Failed to get access token:", data);
    throw new Error("Failed to get Gmail access token");
  }
  return data.access_token;
}

async function sendEmail(accessToken: string, rawEmail: string): Promise<void> {
  const response = await fetch(
    "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ raw: rawEmail }),
    }
  );

  if (!response.ok) {
    const error = await response.text();
    console.error("Gmail API error:", error);
    throw new Error(`Failed to send email: ${error}`);
  }
}

function encodeEmail(to: string, subject: string, htmlBody: string): string {
  const gmailUser = Deno.env.get("GMAIL_USER") || "belmarj@vectismanufacturing.com";
  
  const messageParts = [
    `From: "Vectis Manufacturing" <${gmailUser}>`,
    `To: ${to}`,
    `Subject: ${subject}`,
    `MIME-Version: 1.0`,
    `Content-Type: text/html; charset=utf-8`,
    `Content-Transfer-Encoding: base64`,
    "",
    btoa(unescape(encodeURIComponent(htmlBody))),
  ];
  
  const rawMessage = messageParts.join("\r\n");
  
  const encoder = new TextEncoder();
  const bytes = encoder.encode(rawMessage);

  const chunkSize = 8192;
  let binaryString = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.slice(i, i + chunkSize);
    binaryString += String.fromCharCode(...chunk);
  }

  const base64 = btoa(binaryString);
  return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

// Helper to generate detail rows - same as send-contact-message
function generateDetailRow(label: string, value: string, isMultiline: boolean = false): string {
  if (isMultiline) {
    return `
      <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="border-bottom: 1px solid #e2e8f0;">
        <tr>
          <td style="padding: 12px 15px;">
            <span style="color: #64748b; font-size: 12px; font-weight: 600; display: block; margin-bottom: 8px;">${label}</span>
            <p style="color: #1e293b; font-size: 13px; margin: 0; line-height: 1.6; white-space: pre-line;">${value}</p>
          </td>
        </tr>
      </table>
    `;
  }
  return `
    <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="border-bottom: 1px solid #e2e8f0;">
      <tr>
        <td style="padding: 10px 15px; width: 40%; color: #64748b; font-size: 12px; font-weight: 600; vertical-align: top;">${label}</td>
        <td style="padding: 10px 15px; width: 60%; color: #1e293b; font-size: 12px; font-weight: 600; text-align: right; vertical-align: top;">${value}</td>
      </tr>
    </table>
  `;
}

// Generate section header - same as send-contact-message
function generateSectionHeader(title: string): string {
  return `
    <div style="background-color: rgba(239, 246, 255, 0.9); padding: 12px 20px; border-bottom: 1px solid #dbeafe;">
      <h3 style="color: #1e40af; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin: 0;">${title}</h3>
    </div>
  `;
}

// Email template - same professional design as send-contact-message but with customer-facing content
function generateConfirmationEmailTemplate(options: {
  customerName: string;
  quoteNumber: string;
  projectDetailsContent: string;
  partDetailsContent?: string;
}): string {
  const { customerName, quoteNumber, projectDetailsContent, partDetailsContent } = options;

  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Quote Request Received</title>
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

            <div style="margin: 0 auto; max-width: 600px; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); background-image: url('https://res.cloudinary.com/dbcfeio6b/image/upload/v1765522367/LOGO_-_Copy-removebg-preview_gu9f3c.png'); background-repeat: no-repeat; background-position: center 22%; background-size: 80%;">
              <div style="background-color: rgba(255, 255, 255, 0.93); width: 100%; height: 100%;">
              
                <!-- 1. Brand Header -->
                <div style="background-color: #000000; padding: 20px 15px; text-align: center; position: relative; z-index: 2;">
                  <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%">
                    <tr>
                      <td align="center">
                        <img src="https://res.cloudinary.com/dbcfeio6b/image/upload/v1765508292/output-onlinepngtools-removebg-preview_1_kkhayz.png" alt="VM Logo" width="70" style="display: block; margin: 0 auto 10px auto; height: auto; border: 0;">
                      </td>
                    </tr>
                    <tr>
                      <td align="center">
                        <span style="color: #ffffff; font-size: 20px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">Vectis Manufacturing</span>
                      </td>
                    </tr>
                  </table>
                </div>

                <!-- Hero Section -->
                <div style="padding: 30px 20px 15px 20px; text-align: center; background-color: transparent;">
                  <div style="display: inline-block; width: 56px; height: 56px; border-radius: 50%; background-color: rgba(187, 247, 208, 0.85); margin-bottom: 15px; line-height: 56px;">
                    <span style="font-size: 28px; color: #16a34a; line-height: 56px; font-family: Arial, sans-serif;">&#10003;</span>
                  </div>
                  
                  <h2 style="color: #1e293b; font-size: 20px; font-weight: 700; margin: 0 0 8px 0;">Quote Request Received</h2>
                  <p style="color: #64748b; font-size: 14px; margin: 0; line-height: 1.5;">Thank you, <strong>${customerName}</strong>! We have received your quote request.</p>
                </div>

                <!-- Reference Number -->
                <div style="padding: 0 20px;">
                  <div style="background-color: rgba(240, 253, 244, 0.9); border: 1px solid #bbf7d0; border-radius: 6px; padding: 15px; text-align: center;">
                    <p style="color: #166534; font-size: 14px; margin: 0;">
                      <span style="font-size: 18px; margin-right: 8px;">✅</span>
                      <strong>Your Reference Number:</strong> ${quoteNumber}
                    </p>
                  </div>
                </div>

                <!-- Content & Details -->
                <div style="padding: 0 20px 30px 20px;">

                  <!-- Project Details Card -->
                  <div style="background-color: rgba(248, 250, 252, 0.85); border: 1px solid #e2e8f0; border-radius: 6px; padding: 0; margin-top: 25px; overflow: hidden;">
                    ${generateSectionHeader("Your Details")}
                    ${projectDetailsContent}
                  </div>

                  ${partDetailsContent ? `
                  <!-- Part Details Card -->
                  <div style="background-color: rgba(248, 250, 252, 0.85); border: 1px solid #e2e8f0; border-radius: 6px; padding: 0; margin-top: 20px; overflow: hidden;">
                    ${generateSectionHeader("Part Details")}
                    ${partDetailsContent}
                  </div>
                  ` : ""}

                  <!-- What Happens Next -->
                  <div style="margin-top: 30px; text-align: center; padding: 20px; background-color: rgba(239, 246, 255, 0.9); border: 1px solid #bfdbfe; border-radius: 6px;">
                    <p style="color: #1e40af; font-size: 14px; font-weight: 600; margin: 0 0 10px 0;">
                      What happens next?
                    </p>
                    <p style="color: #1e40af; font-size: 13px; margin: 0; line-height: 1.6;">
                      Our engineering team will review your CAD files and specifications.<br>
                      You can expect to receive a detailed quote within <strong>1-2 business days</strong>.
                    </p>
                  </div>

                  <p style="text-align: center; color: #64748b; font-size: 14px; margin-top: 30px;">
                    If you have any questions, please contact us at<br>
                    <a href="mailto:belmarj@vectismanufacturing.com" style="color: #2563eb; text-decoration: none;">belmarj@vectismanufacturing.com</a>
                  </p>

                </div>

              </div>
            </div>

            <!-- Footer -->
            <div style="background-color: #f1f4f9; padding: 30px; text-align: center; font-size: 12px; color: #94a3b8;">
              <p style="margin-bottom: 10px;">&copy; ${new Date().getFullYear()} Vectis Manufacturing. All rights reserved.</p>
              <p style="margin: 0;">Precision CNC Machining & Custom Manufacturing</p>
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

const handler = async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { 
      customerName, 
      customerEmail, 
      quoteNumber, 
      company,
      phone,
      address,
      projectDescription,
      partDetails 
    }: ConfirmationRequest = await req.json();

    console.log("Sending quote request confirmation to:", customerEmail);

    // Generate Project Details content
    let projectDetailsContent = `
      ${generateDetailRow("Name", customerName)}
      ${company ? generateDetailRow("Company", company) : ""}
      ${generateDetailRow("Email", customerEmail)}
      ${phone ? generateDetailRow("Phone", phone) : ""}
      ${address ? generateDetailRow("Address", address) : ""}
      ${projectDescription ? generateDetailRow("Project Description", projectDescription, true) : ""}
    `;

    // Generate Part Details content if provided
    let partDetailsContent = "";
    if (partDetails) {
      const rows = [];
      if (partDetails.partName) rows.push(generateDetailRow("Part / Job Name", partDetails.partName));
      if (partDetails.material) rows.push(generateDetailRow("Material", partDetails.material));
      if (partDetails.quantity) rows.push(generateDetailRow("Quantity", String(partDetails.quantity)));
      if (partDetails.finish) rows.push(generateDetailRow("Finish", partDetails.finish));
      if (partDetails.heatTreatment) {
        rows.push(generateDetailRow("Heat Treatment", "Required"));
        if (partDetails.heatTreatmentDetails) {
          rows.push(generateDetailRow("Heat Treatment Details", partDetails.heatTreatmentDetails));
        }
      }
      if (partDetails.threadsTolerances) rows.push(generateDetailRow("Threads / Tolerances", partDetails.threadsTolerances, true));
      
      if (rows.length > 0) {
        partDetailsContent = rows.join("");
      }
    }

    // Generate email HTML using the same professional template
    const htmlBody = generateConfirmationEmailTemplate({
      customerName,
      quoteNumber,
      projectDetailsContent,
      partDetailsContent: partDetailsContent || undefined,
    });
    
    // Get access token and send email
    const accessToken = await getAccessToken();
    const rawEmail = encodeEmail(
      customerEmail,
      `Quote Request Received - ${quoteNumber}`,
      htmlBody
    );
    
    await sendEmail(accessToken, rawEmail);

    console.log("Confirmation email sent successfully to:", customerEmail);

    return new Response(
      JSON.stringify({ success: true, message: "Confirmation email sent" }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error: any) {
    console.error("Error sending confirmation email:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
};

serve(handler);