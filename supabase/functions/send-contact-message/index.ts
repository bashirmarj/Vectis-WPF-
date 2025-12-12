import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const supabase = createClient(Deno.env.get("SUPABASE_URL") ?? "", Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "");

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Helper to get OAuth2 access token using refresh token
async function getAccessToken(): Promise<string> {
  const response = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      client_id: Deno.env.get("GMAIL_CLIENT_ID")!,
      client_secret: Deno.env.get("GMAIL_CLIENT_SECRET")!,
      refresh_token: Deno.env.get("GMAIL_REFRESH_TOKEN")!,
    }),
  });
  const data = await response.json();
  if (data.error) {
    throw new Error(`OAuth error: ${data.error_description || data.error}`);
  }
  return data.access_token;
}

// Helper to send email via Gmail API
async function sendEmail(accessToken: string, rawEmail: string): Promise<void> {
  const response = await fetch("https://gmail.googleapis.com/gmail/v1/users/me/messages/send", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ raw: rawEmail }),
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Gmail API error: ${error}`);
  }
}

// Helper to encode email in base64url format
function encodeEmail(to: string, subject: string, htmlBody: string, replyTo?: string): string {
  const gmailUser = Deno.env.get("GMAIL_USER") || "belmarj@vectismanufacturing.com";
  const boundary = `boundary_${Date.now()}`;

  const messageParts = [
    `From: "Vectis Manufacturing" <${gmailUser}>`,
    `To: ${to}`,
    `Subject: ${subject}`,
    replyTo ? `Reply-To: ${replyTo}` : "",
    `MIME-Version: 1.0`,
    `Content-Type: multipart/alternative; boundary="${boundary}"`,
    "",
    `--${boundary}`,
    `Content-Type: text/html; charset=utf-8`,
    `Content-Transfer-Encoding: base64`,
    "",
    btoa(unescape(encodeURIComponent(htmlBody))),
    `--${boundary}--`,
  ]
    .filter(Boolean)
    .join("\r\n");

  // Convert to base64url (Gmail API requirement)
  const encoder = new TextEncoder();
  const bytes = encoder.encode(messageParts);
  const base64 = btoa(String.fromCharCode(...bytes));
  return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

// Helper function to hash IP addresses for privacy
async function hashIP(ip: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(ip);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

interface ContactRequest {
  name: string;
  email: string;
  phone?: string;
  company?: string;
  message: string;
}

// Unified email template generator for contact messages
function generateContactEmailTemplate(options: {
  heroTitle: string;
  heroSubtitle: string;
  detailsContent: string;
  footerText?: string;
}): string {
  const { heroTitle, heroSubtitle, detailsContent, footerText } = options;

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

            <div style="margin: 0 auto; max-width: 600px; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); background-image: url('https://res.cloudinary.com/dbcfeio6b/image/upload/v1765522367/LOGO_-_Copy-removebg-preview_gu9f3c.png'); background-repeat: no-repeat; background-position: center 22%; background-size: 80%;">
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

                <!-- 3. Hero Section -->
                <div style="padding: 40px 40px 20px 40px; text-align: center; background-color: transparent;">
                  <div style="display: inline-block; width: 64px; height: 64px; border-radius: 50%; background-color: rgba(224, 242, 254, 0.85); margin-bottom: 20px; line-height: 64px;">
                    <span style="font-size: 32px; color: #0284c7; line-height: 64px; font-family: Arial, sans-serif;">&#9993;</span>
                  </div>
                  
                  <h2 style="color: #1e293b; font-size: 22px; font-weight: 700; margin: 0 0 10px 0;">${heroTitle}</h2>
                  <p style="color: #64748b; font-size: 16px; margin: 0; line-height: 1.5;">${heroSubtitle}</p>
                </div>

                <!-- 4. Content & Details -->
                <div style="padding: 0 40px 40px 40px;">

                  <!-- Details "Receipt" Card -->
                  <div style="background-color: rgba(248, 250, 252, 0.85); border: 1px solid #e2e8f0; border-radius: 6px; padding: 0; margin-top: 25px; overflow: hidden;">
                    <div style="background-color: rgba(239, 246, 255, 0.9); padding: 12px 20px; border-bottom: 1px solid #dbeafe;">
                      <h3 style="color: #1e40af; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin: 0;">Contact Details</h3>
                    </div>
                    
                    ${detailsContent}
                  </div>

                  <!-- Response Note -->
                  <div style="margin-top: 30px; text-align: center; padding: 20px; background-color: rgba(255, 251, 235, 0.9); border: 1px solid #fcd34d; border-radius: 6px;">
                    <p style="color: #92400e; font-size: 14px; font-weight: 500; margin: 0;">
                      &#9201; Please respond within <strong>24-48 hours</strong>
                    </p>
                  </div>

                </div>

              </div>
            </div>

            <!-- Footer -->
            <div style="background-color: #f1f4f9; padding: 30px; text-align: center; font-size: 12px; color: #94a3b8;">
              <p style="margin-bottom: 10px;">&copy; ${new Date().getFullYear()} Vectis Manufacturing. All rights reserved.</p>
              ${footerText ? `<p>${footerText}</p>` : ""}
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
function generateDetailRow(label: string, value: string, isMultiline: boolean = false): string {
  if (isMultiline) {
    return `
      <div style="padding: 12px 20px; border-bottom: 1px solid #e2e8f0;">
        <span style="color: #64748b; font-size: 13px; font-weight: 600; display: block; margin-bottom: 8px;">${label}</span>
        <p style="color: #1e293b; font-size: 14px; margin: 0; line-height: 1.6; white-space: pre-line;">${value}</p>
      </div>
    `;
  }
  return `
    <div style="padding: 12px 20px; border-bottom: 1px solid #e2e8f0;">
      <span style="color: #64748b; font-size: 13px; font-weight: 600; float: left; width: 30%;">${label}</span>
      <span style="color: #1e293b; font-size: 13px; font-weight: 600; float: right; width: 70%; text-align: right;">${value}</span>
      <div style="clear: both;"></div>
    </div>
  `;
}

const handler = async (req: Request): Promise<Response> => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Extract client IP address (rate limiting disabled for now)
    const clientIp =
      req.headers.get("x-forwarded-for")?.split(",")[0].trim() || req.headers.get("x-real-ip") || "unknown";

    console.log("Contact form submission from IP:", clientIp);

    // Hash the IP for privacy
    const ipHash = await hashIP(clientIp);

    // Parse request body
    const { name, email, phone, company, message }: ContactRequest = await req.json();

    console.log("Processing contact form from:", email);

    // Generate details content
    const detailsContent = `
      ${generateDetailRow("Name", name)}
      ${company ? generateDetailRow("Company", company) : ""}
      ${generateDetailRow("Email", email)}
      ${phone ? generateDetailRow("Phone", phone) : ""}
      ${generateDetailRow("Message", message, true)}
    `;

    // Generate email HTML using unified template
    const emailHtml = generateContactEmailTemplate({
      heroTitle: "New Contact Message",
      heroSubtitle: `You have received a new message from <strong>${name}</strong>.`,
      detailsContent: detailsContent,
      footerText: "This message was sent via the website contact form.",
    });

    // Get access token and send email
    const gmailUser = Deno.env.get("GMAIL_USER") || "belmarj@vectismanufacturing.com";
    const accessToken = await getAccessToken();

    const encodedMessage = encodeEmail(gmailUser, `New Contact Message from ${name}`, emailHtml, email);

    await sendEmail(accessToken, encodedMessage);

    console.log("Email sent successfully via Gmail API");

    // Record the submission for rate limiting
    const { error: insertError } = await supabase.from("contact_submissions").insert({
      ip_hash: ipHash,
      email: email,
      submitted_at: new Date().toISOString(),
    });

    if (insertError) {
      console.error("Error recording submission:", insertError);
      // Don't fail the request if we can't record the submission
    }

    return new Response(JSON.stringify({ success: true, message: "Message sent successfully" }), {
      status: 200,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  } catch (error: any) {
    console.error("Error in send-contact-message function:", error);
    return new Response(JSON.stringify({ error: error.message || "Internal server error" }), {
      status: 500,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  }
};

serve(handler);
