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

  let messageParts = [
    `From: "Vectis Manufacturing" <${gmailUser}>`,
    `To: ${to}`,
    `Subject: ${subject}`,
    `MIME-Version: 1.0`,
  ];
  
  if (replyTo) {
    messageParts.push(`Reply-To: ${replyTo}`);
  }
  
  messageParts.push(`Content-Type: text/html; charset=utf-8`);
  messageParts.push("Content-Transfer-Encoding: base64");
  messageParts.push("");
  messageParts.push(btoa(unescape(encodeURIComponent(htmlBody))));

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
  message?: string; // Legacy field for simple contact form
}

// Helper to generate detail rows
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

// Generate section header
function generateSectionHeader(title: string): string {
  return `
    <div style="background-color: rgba(239, 246, 255, 0.9); padding: 12px 20px; border-bottom: 1px solid #dbeafe;">
      <h3 style="color: #1e40af; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin: 0;">${title}</h3>
    </div>
  `;
}

// Email template generator for contact/quote messages
function generateContactEmailTemplate(options: {
  heroTitle: string;
  heroSubtitle: string;
  projectDetailsContent: string;
  partDetailsContent?: string;
  footerText?: string;
}): string {
  const { heroTitle, heroSubtitle, projectDetailsContent, partDetailsContent, footerText } = options;

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
                  <div style="display: inline-block; width: 56px; height: 56px; border-radius: 50%; background-color: rgba(224, 242, 254, 0.85); margin-bottom: 15px; line-height: 56px;">
                    <span style="font-size: 28px; color: #0284c7; line-height: 56px; font-family: Arial, sans-serif;">&#9993;</span>
                  </div>
                  
                  <h2 style="color: #1e293b; font-size: 20px; font-weight: 700; margin: 0 0 8px 0;">${heroTitle}</h2>
                  <p style="color: #64748b; font-size: 14px; margin: 0; line-height: 1.5;">${heroSubtitle}</p>
                </div>

                <!-- Content & Details -->
                <div style="padding: 0 20px 30px 20px;">

                  <!-- Project Details Card -->
                  <div style="background-color: rgba(248, 250, 252, 0.85); border: 1px solid #e2e8f0; border-radius: 6px; padding: 0; margin-top: 25px; overflow: hidden;">
                    ${generateSectionHeader("Project Details")}
                    ${projectDetailsContent}
                  </div>

                  ${partDetailsContent ? `
                  <!-- Part Details Card -->
                  <div style="background-color: rgba(248, 250, 252, 0.85); border: 1px solid #e2e8f0; border-radius: 6px; padding: 0; margin-top: 20px; overflow: hidden;">
                    ${generateSectionHeader("Part Details")}
                    ${partDetailsContent}
                  </div>
                  ` : ""}

                  <!-- Response Note -->
                  <div style="margin-top: 30px; text-align: center; padding: 20px; background-color: rgba(255, 251, 235, 0.9); border: 1px solid #fcd34d; border-radius: 6px;">
                    <p style="color: #92400e; font-size: 14px; font-weight: 500; margin: 0;">
                      &#9201; Please respond within <strong>24-48 hours</strong>
                    </p>
                  </div>

                  <p style="text-align: center; color: #64748b; font-size: 14px; margin-top: 30px;">
                    Reply directly to this email to respond to the sender
                  </p>

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

// Input validation helpers
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MAX_NAME_LENGTH = 100;
const MAX_EMAIL_LENGTH = 255;
const MAX_PHONE_LENGTH = 30;
const MAX_COMPANY_LENGTH = 200;
const MAX_MESSAGE_LENGTH = 5000;

function validateContactInput(data: any): { valid: boolean; error?: string } {
  if (!data || typeof data !== 'object') {
    return { valid: false, error: 'Invalid request body' };
  }
  
  const { name, email } = data;
  
  // Required fields
  if (!name || typeof name !== 'string' || name.trim().length === 0) {
    return { valid: false, error: 'Name is required' };
  }
  if (!email || typeof email !== 'string' || email.trim().length === 0) {
    return { valid: false, error: 'Email is required' };
  }
  
  // Length limits
  if (name.length > MAX_NAME_LENGTH) {
    return { valid: false, error: `Name must be less than ${MAX_NAME_LENGTH} characters` };
  }
  if (email.length > MAX_EMAIL_LENGTH) {
    return { valid: false, error: `Email must be less than ${MAX_EMAIL_LENGTH} characters` };
  }
  if (data.phone && data.phone.length > MAX_PHONE_LENGTH) {
    return { valid: false, error: `Phone must be less than ${MAX_PHONE_LENGTH} characters` };
  }
  if (data.company && data.company.length > MAX_COMPANY_LENGTH) {
    return { valid: false, error: `Company must be less than ${MAX_COMPANY_LENGTH} characters` };
  }
  
  // Email format
  if (!EMAIL_REGEX.test(email)) {
    return { valid: false, error: 'Invalid email format' };
  }
  
  return { valid: true };
}

// Sanitize text to prevent XSS
function sanitizeText(text: string): string {
  return text.replace(/[<>]/g, '').trim();
}

const handler = async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const clientIp =
      req.headers.get("x-forwarded-for")?.split(",")[0].trim() || req.headers.get("x-real-ip") || "unknown";

    console.log("Contact form submission from IP:", clientIp);

    const ipHash = await hashIP(clientIp);

    const rawBody = await req.json();
    const validation = validateContactInput(rawBody);
    
    if (!validation.valid) {
      console.warn("Input validation failed:", validation.error);
      return new Response(
        JSON.stringify({ error: validation.error }),
        { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
      );
    }
    
    // Sanitize inputs
    const name = sanitizeText(rawBody.name);
    const email = rawBody.email.trim().toLowerCase();
    const phone = rawBody.phone ? sanitizeText(rawBody.phone) : undefined;
    const company = rawBody.company ? sanitizeText(rawBody.company) : undefined;
    const address = rawBody.address ? sanitizeText(rawBody.address) : undefined;
    const projectDescription = rawBody.projectDescription ? sanitizeText(rawBody.projectDescription) : undefined;
    const message = rawBody.message ? sanitizeText(rawBody.message) : undefined;
    const partDetails = rawBody.partDetails;

    console.log("Processing contact form from:", email);

    // Generate Project Details content
    let projectDetailsContent = `
      ${generateDetailRow("Name", name)}
      ${company ? generateDetailRow("Company", company) : ""}
      ${generateDetailRow("Email", email)}
      ${phone ? generateDetailRow("Phone", phone) : ""}
      ${address ? generateDetailRow("Address", address) : ""}
      ${projectDescription ? generateDetailRow("Project Description", projectDescription, true) : ""}
      ${message ? generateDetailRow("Message", message, true) : ""}
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

    // Generate email HTML
    const emailHtml = generateContactEmailTemplate({
      heroTitle: "New Quote Request",
      heroSubtitle: `You have received a new quote request from <strong>${name}</strong>.`,
      projectDetailsContent: projectDetailsContent,
      partDetailsContent: partDetailsContent || undefined,
      footerText: "This message was sent via the website quote request form.",
    });

    // Get access token and send email
    const gmailUser = Deno.env.get("GMAIL_USER") || "belmarj@vectismanufacturing.com";
    const accessToken = await getAccessToken();

    const encodedMessage = encodeEmail(gmailUser, `New Quote Request from ${name}`, emailHtml, email);

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