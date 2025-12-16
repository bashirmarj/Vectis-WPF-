import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(supabaseUrl, supabaseServiceKey);

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

// Helper to encode email with attachments in base64url format (MIME multipart)
function encodeEmailWithAttachments(
  to: string,
  subject: string,
  htmlBody: string,
  attachments?: Array<{ filename: string; content: string }>,
): string {
  const gmailUser = Deno.env.get("GMAIL_USER") || "belmarj@vectismanufacturing.com";
  const boundary = `boundary_${Date.now()}`;

  let messageParts = [
    `From: "Vectis Manufacturing" <${gmailUser}>`,
    `To: ${to}`,
    `Subject: ${subject}`,
    `MIME-Version: 1.0`,
  ];

  if (attachments && attachments.length > 0) {
    // Multipart mixed for email with attachments
    messageParts.push(`Content-Type: multipart/mixed; boundary="${boundary}"`);
    messageParts.push("");

    // HTML body part
    messageParts.push(`--${boundary}`);
    messageParts.push("Content-Type: text/html; charset=utf-8");
    messageParts.push("Content-Transfer-Encoding: base64");
    messageParts.push("");
    messageParts.push(btoa(unescape(encodeURIComponent(htmlBody))));

    // Attachment parts
    for (const attachment of attachments) {
      const mimeType = getMimeType(attachment.filename);
      messageParts.push(`--${boundary}`);
      messageParts.push(`Content-Type: ${mimeType}; name="${attachment.filename}"`);
      messageParts.push("Content-Transfer-Encoding: base64");
      messageParts.push(`Content-Disposition: attachment; filename="${attachment.filename}"`);
      messageParts.push("");
      messageParts.push(attachment.content); // Already base64 encoded from client
    }

    messageParts.push(`--${boundary}--`);
  } else {
    // Simple email without attachments
    messageParts.push(`Content-Type: text/html; charset=utf-8`);
    messageParts.push("Content-Transfer-Encoding: base64");
    messageParts.push("");
    messageParts.push(btoa(unescape(encodeURIComponent(htmlBody))));
  }

  const rawMessage = messageParts.join("\r\n");

  // Convert to base64url (Gmail API requirement)
  const encoder = new TextEncoder();
  const bytes = encoder.encode(rawMessage);

  // Process in chunks to avoid stack overflow with large attachments
  const chunkSize = 8192;
  let binaryString = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.slice(i, i + chunkSize);
    binaryString += String.fromCharCode(...chunk);
  }

  const base64 = btoa(binaryString);
  return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

// Helper to get MIME type based on file extension
function getMimeType(filename: string): string {
  const ext = filename.toLowerCase().split(".").pop();
  const mimeTypes: Record<string, string> = {
    step: "application/STEP",
    stp: "application/STEP",
    stl: "application/sla",
    obj: "model/obj",
    iges: "model/iges",
    igs: "model/iges",
    pdf: "application/pdf",
    dxf: "application/dxf",
    dwg: "application/dwg",
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
  };
  return mimeTypes[ext || ""] || "application/octet-stream";
}

interface FileInfo {
  name: string;
  content: string;
  size: number;
  quantity: number;
  material?: string;
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
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Helper function to hash strings
async function hashString(input: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
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
    footerText,
  } = options;

  const statusTracker = showStatusTracker
    ? `
    <!-- 2. Visual Status Tracker - Table-based for mobile -->
    <div style="background-color: rgba(248, 250, 252, 0.9); padding: 20px 10px; border-bottom: 1px solid #e2e8f0;">
      <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%">
        <tr>
          <td width="33%" align="center" valign="top" style="padding: 5px;">
            <span style="height: 12px; width: 12px; background-color: ${statusStep >= 1 ? "#10b981" : "#cbd5e1"}; border-radius: 50%; display: inline-block; margin-bottom: 8px; ${statusStep >= 1 ? "box-shadow: 0 0 0 4px #d1fae5;" : ""}"></span>
            <br>
            <span style="font-size: 10px; color: ${statusStep >= 1 ? "#10b981" : "#64748b"}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Received</span>
          </td>
          <td width="33%" align="center" valign="top" style="padding: 5px;">
            <span style="height: 12px; width: 12px; background-color: ${statusStep >= 2 ? "#10b981" : "#cbd5e1"}; border-radius: 50%; display: inline-block; margin-bottom: 8px; ${statusStep >= 2 ? "box-shadow: 0 0 0 4px #d1fae5;" : ""}"></span>
            <br>
            <span style="font-size: 10px; color: ${statusStep >= 2 ? "#10b981" : "#64748b"}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Reviewing</span>
          </td>
          <td width="33%" align="center" valign="top" style="padding: 5px;">
            <span style="height: 12px; width: 12px; background-color: ${statusStep >= 3 ? "#10b981" : "#cbd5e1"}; border-radius: 50%; display: inline-block; margin-bottom: 8px; ${statusStep >= 3 ? "box-shadow: 0 0 0 4px #d1fae5;" : ""}"></span>
            <br>
            <span style="font-size: 10px; color: ${statusStep >= 3 ? "#10b981" : "#64748b"}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Quote Ready</span>
          </td>
        </tr>
      </table>
    </div>
  `
    : "";

  const timelineBox = timelineText
    ? `
    <!-- Timeline / Next Steps -->
    <div style="margin-top: 30px; text-align: center; padding: 20px; background-color: rgba(255, 251, 235, 0.9); border: 1px solid #fcd34d; border-radius: 6px;">
      <p style="color: #92400e; font-size: 14px; font-weight: 500; margin: 0;">
        &#9201; ${timelineText}
      </p>
    </div>
  `
    : "";

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
              
                <!-- 1. Brand Header - Table-based for mobile -->
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

                ${statusTracker}

                <!-- 3. Hero Section -->
                <div style="padding: 30px 20px 15px 20px; text-align: center; background-color: transparent;">
                  <div style="display: inline-block; width: 56px; height: 56px; border-radius: 50%; background-color: rgba(224, 242, 254, 0.85); margin-bottom: 15px; line-height: 56px;">
                    <span style="font-size: 28px; color: #0284c7; line-height: 56px; font-family: Arial, sans-serif;">&#10003;</span>
                  </div>
                  
                  <h2 style="color: #1e293b; font-size: 20px; font-weight: 700; margin: 0 0 8px 0;">${heroTitle}</h2>
                  <p style="color: #64748b; font-size: 14px; margin: 0; line-height: 1.5;">${heroSubtitle}</p>
                </div>

                <!-- Reference Number Block -->
                <div style="text-align: center; padding-bottom: 25px; background-color: transparent;">
                  <span style="background: rgba(226, 232, 240, 0.85); color: #475569; padding: 6px 12px; border-radius: 15px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px;">REF: ${quoteNumber}</span>
                </div>

                <!-- 4. Content & Details -->
                <div style="padding: 0 20px 30px 20px;">

                  <!-- Details "Receipt" Card -->
                  <div style="background-color: rgba(248, 250, 252, 0.85); border: 1px solid #e2e8f0; border-radius: 6px; padding: 0; margin-top: 25px; overflow: hidden;">
                    <div style="background-color: rgba(239, 246, 255, 0.9); padding: 12px 20px; border-bottom: 1px solid #dbeafe;">
                      <h3 style="color: #1e40af; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin: 0;">Order Summary</h3>
                    </div>
                    
                    ${detailsContent}
                    
                    ${fileListContent || ""}
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

// Helper to generate detail rows - Table-based for mobile
function generateDetailRow(label: string, value: string): string {
  return `
    <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="border-bottom: 1px solid #e2e8f0;">
      <tr>
        <td style="padding: 10px 15px; width: 40%; color: #64748b; font-size: 12px; font-weight: 600; vertical-align: top;">${label}</td>
        <td style="padding: 10px 15px; width: 60%; color: #1e293b; font-size: 12px; font-weight: 600; text-align: right; vertical-align: top;">${value}</td>
      </tr>
    </table>
  `;
}

// Helper to generate part section header
function generatePartHeader(partNumber: number): string {
  return `
    <div style="background-color: rgba(239, 246, 255, 0.9); padding: 10px 15px; border-top: 2px solid #3b82f6; margin-top: 8px;">
      <span style="color: #1e40af; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Part ${partNumber}</span>
    </div>
  `;
}

// Helper to parse and format part details from message string
function parseAndFormatPartDetails(message: string): string {
  if (!message) return '';
  
  // Split by "--- Part Details ---" to separate parts
  const parts = message.split(/---\s*Part Details\s*---/i);
  
  let formattedHtml = '';
  
  // First part may contain project-level info (before any Part Details)
  const projectInfo = parts[0].trim();
  if (projectInfo && projectInfo.length > 0) {
    // Parse project info lines
    const projectLines = projectInfo.split('\n');
    for (const line of projectLines) {
      const colonIndex = line.indexOf(':');
      if (colonIndex > 0) {
        const label = line.substring(0, colonIndex).trim();
        const value = line.substring(colonIndex + 1).trim();
        if (label && value) {
          formattedHtml += generateDetailRow(label, value);
        }
      }
    }
  }
  
  // Each subsequent part is a part's details
  for (let i = 1; i < parts.length; i++) {
    const partBlock = parts[i].trim();
    if (!partBlock) continue;
    
    // Add a part header
    formattedHtml += generatePartHeader(i);
    
    // Parse each line (e.g., "Part Name: Main body Rev01")
    const lines = partBlock.split('\n');
    for (const line of lines) {
      const trimmedLine = line.trim();
      if (!trimmedLine) continue;
      
      const colonIndex = trimmedLine.indexOf(':');
      if (colonIndex > 0) {
        const label = trimmedLine.substring(0, colonIndex).trim();
        const value = trimmedLine.substring(colonIndex + 1).trim();
        if (label && value) {
          formattedHtml += generateDetailRow(label, value);
        }
      }
    }
  }
  
  return formattedHtml;
}

// Helper to generate file list - Table-based for mobile
function generateFileList(files: FileInfo[], drawingFiles?: FileInfo[]): string {
  const fileItems = files
    .map(
      (f) => `
    <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: rgba(255, 255, 255, 0.8); border: 1px solid #e2e8f0; border-radius: 4px; margin-top: 8px;">
      <tr>
        <td style="padding: 10px; width: 60%; vertical-align: middle;">
          <span style="font-size: 13px; color: #334155; font-weight: 500; word-break: break-word;">${f.name}</span>
        </td>
        <td style="padding: 10px; width: 40%; text-align: right; vertical-align: middle;">
          <span style="font-size: 11px; color: #3b82f6; font-weight: 600;">${f.material || "TBD"}</span>
          <span style="font-size: 11px; color: #64748b; font-weight: 600; margin-left: 8px;">x${f.quantity}</span>
        </td>
      </tr>
    </table>
  `,
    )
    .join("");

  const drawingItems =
    drawingFiles && drawingFiles.length > 0
      ? drawingFiles
          .map(
            (f) => `
    <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: rgba(255, 255, 255, 0.8); border: 1px solid #e2e8f0; border-radius: 4px; margin-top: 8px;">
      <tr>
        <td style="padding: 10px; width: 70%; vertical-align: middle;">
          <span style="font-size: 13px; color: #334155; font-weight: 500; word-break: break-word;">${f.name}</span>
        </td>
        <td style="padding: 10px; width: 30%; text-align: right; vertical-align: middle;">
          <span style="font-size: 11px; color: #64748b; font-weight: 600;">Drawing</span>
        </td>
      </tr>
    </table>
  `,
          )
          .join("")
      : "";

  return `
    <div style="padding: 10px 15px; background-color: rgba(255,255,255,0.7);">
      <span style="color: #64748b; font-size: 12px; font-weight: 600; width: 100%; margin-bottom: 6px; display: block;">Uploaded Files</span>
      ${fileItems}
      ${drawingItems}
    </div>
  `;
}

// Input validation helpers
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MAX_NAME_LENGTH = 100;
const MAX_EMAIL_LENGTH = 255;
const MAX_PHONE_LENGTH = 30;
const MAX_COMPANY_LENGTH = 200;
const MAX_ADDRESS_LENGTH = 500;
const MAX_MESSAGE_LENGTH = 5000;
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB per file
const ALLOWED_EXTENSIONS = ['step', 'stp', 'iges', 'igs', 'stl', 'obj', 'pdf', 'dxf', 'dwg', 'png', 'jpg', 'jpeg'];

function validateQuotationInput(data: any): { valid: boolean; error?: string } {
  if (!data || typeof data !== 'object') {
    return { valid: false, error: 'Invalid request body' };
  }
  
  const { name, email, phone, files } = data;
  
  // Required fields
  if (!name || typeof name !== 'string' || name.trim().length === 0) {
    return { valid: false, error: 'Name is required' };
  }
  if (!email || typeof email !== 'string' || email.trim().length === 0) {
    return { valid: false, error: 'Email is required' };
  }
  if (!phone || typeof phone !== 'string' || phone.trim().length === 0) {
    return { valid: false, error: 'Phone is required' };
  }
  if (!files || !Array.isArray(files) || files.length === 0) {
    return { valid: false, error: 'At least one file is required' };
  }
  
  // Length limits
  if (name.length > MAX_NAME_LENGTH) {
    return { valid: false, error: `Name must be less than ${MAX_NAME_LENGTH} characters` };
  }
  if (email.length > MAX_EMAIL_LENGTH) {
    return { valid: false, error: `Email must be less than ${MAX_EMAIL_LENGTH} characters` };
  }
  if (phone.length > MAX_PHONE_LENGTH) {
    return { valid: false, error: `Phone must be less than ${MAX_PHONE_LENGTH} characters` };
  }
  if (data.company && data.company.length > MAX_COMPANY_LENGTH) {
    return { valid: false, error: `Company must be less than ${MAX_COMPANY_LENGTH} characters` };
  }
  if (data.shippingAddress && data.shippingAddress.length > MAX_ADDRESS_LENGTH) {
    return { valid: false, error: `Shipping address must be less than ${MAX_ADDRESS_LENGTH} characters` };
  }
  if (data.message && data.message.length > MAX_MESSAGE_LENGTH) {
    return { valid: false, error: `Message must be less than ${MAX_MESSAGE_LENGTH} characters` };
  }
  
  // Email format
  if (!EMAIL_REGEX.test(email)) {
    return { valid: false, error: 'Invalid email format' };
  }
  
  // Validate files
  for (const file of files) {
    if (!file.name || typeof file.name !== 'string') {
      return { valid: false, error: 'Invalid file structure' };
    }
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!ext || !ALLOWED_EXTENSIONS.includes(ext)) {
      return { valid: false, error: `File type .${ext} is not allowed. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}` };
    }
    if (file.size && file.size > MAX_FILE_SIZE) {
      return { valid: false, error: `File ${file.name} exceeds maximum size of 50MB` };
    }
  }
  
  return { valid: true };
}

// Sanitize text to prevent XSS when displayed in admin UI
function sanitizeText(text: string): string {
  return text.replace(/[<>]/g, '').trim();
}

const handler = async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Parse request body
    const rawBody = await req.json();
    
    // Handle notification-only mode (called from QuoteRequestForm after DB insert)
    if (rawBody.notificationOnly) {
      console.log("Processing notification-only email request");
      
      const { name, company, email, phone, address, message, quoteNumber, files = [] } = rawBody;
      
      const sendNotificationEmails = async () => {
        try {
          const accessToken = await getAccessToken();
          const gmailUser = Deno.env.get("GMAIL_USER") || "belmarj@vectismanufacturing.com";

          // Download files from storage and prepare attachments for admin email
          const attachments: Array<{ filename: string; content: string }> = [];
          
          for (const file of files) {
            if (file.path) {
              try {
                console.log(`Downloading file from storage: ${file.path}`);
                const { data, error } = await supabase.storage
                  .from('cad-files')
                  .download(file.path);
                
                if (data && !error) {
                  const buffer = await data.arrayBuffer();
                  const bytes = new Uint8Array(buffer);
                  
                  // Convert to base64 in chunks to avoid stack overflow
                  const chunkSize = 8192;
                  let binaryString = "";
                  for (let i = 0; i < bytes.length; i += chunkSize) {
                    const chunk = bytes.slice(i, i + chunkSize);
                    binaryString += String.fromCharCode(...chunk);
                  }
                  const base64Content = btoa(binaryString);
                  
                  attachments.push({
                    filename: file.name,
                    content: base64Content
                  });
                  console.log(`File downloaded and encoded: ${file.name}`);
                } else {
                  console.error(`Failed to download file ${file.path}:`, error);
                }
              } catch (downloadError) {
                console.error(`Error downloading file ${file.path}:`, downloadError);
              }
            }
          }

          // Parse and format part details from message
          const parsedPartDetails = message ? parseAndFormatPartDetails(message) : '';

          // Generate admin email content with all form details
          const adminDetailsContent = `
            ${generateDetailRow("Quote Number", quoteNumber || "Pending")}
            ${generateDetailRow("Date", new Date().toLocaleDateString())}
            ${generateDetailRow("Name", name)}
            ${generateDetailRow("Company", company || "N/A")}
            ${generateDetailRow("Email", email)}
            ${generateDetailRow("Phone", phone || "N/A")}
            ${address ? generateDetailRow("Shipping Address", address) : ""}
            ${parsedPartDetails}
          `;

          // Simple file list for notification
          const fileListHtml = files.length > 0 
            ? `<div style="padding: 10px 15px; background-color: rgba(255,255,255,0.7);">
                <span style="color: #64748b; font-size: 12px; font-weight: 600; width: 100%; margin-bottom: 6px; display: block;">Uploaded Files</span>
                ${files.map((f: any) => `
                  <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: rgba(255, 255, 255, 0.8); border: 1px solid #e2e8f0; border-radius: 4px; margin-top: 8px;">
                    <tr>
                      <td style="padding: 10px; width: 70%; vertical-align: middle;">
                        <span style="font-size: 13px; color: #334155; font-weight: 500; word-break: break-word;">${f.name}</span>
                      </td>
                      <td style="padding: 10px; width: 30%; text-align: right; vertical-align: middle;">
                        <span style="font-size: 11px; color: #64748b; font-weight: 600;">${(f.size / 1024).toFixed(1)} KB</span>
                      </td>
                    </tr>
                  </table>
                `).join("")}
              </div>`
            : "";

          const adminEmailHtml = generateUnifiedEmailTemplate({
            heroTitle: "New Quote Request",
            heroSubtitle: `A new quotation request has been submitted by <strong>${name}</strong>.<br>Please review the details below.`,
            quoteNumber: quoteNumber || "Pending",
            statusStep: 1,
            detailsContent: adminDetailsContent,
            fileListContent: fileListHtml,
            timelineText: `Review and respond within <strong>24-48 Hours</strong>`,
            footerText: `${attachments.length} file(s) attached to this email.`,
          });

          // Generate customer email with full details (same as admin)
          const customerDetailsContent = `
            ${generateDetailRow("Quote Number", quoteNumber || "Pending")}
            ${generateDetailRow("Date", new Date().toLocaleDateString())}
            ${generateDetailRow("Name", name)}
            ${generateDetailRow("Company", company || "N/A")}
            ${generateDetailRow("Email", email)}
            ${generateDetailRow("Phone", phone || "N/A")}
            ${address ? generateDetailRow("Shipping Address", address) : ""}
            ${parsedPartDetails}
          `;

          const customerEmailHtml = generateUnifiedEmailTemplate({
            heroTitle: "Request Received",
            heroSubtitle: `Thanks for your request, <strong>${name}</strong>.<br>Our engineering team has started reviewing your files.`,
            quoteNumber: quoteNumber || "Pending",
            statusStep: 1,
            detailsContent: customerDetailsContent,
            fileListContent: fileListHtml,
            timelineText: `Estimated Response Time: <strong>24-48 Hours</strong>`,
          });

          // Send admin email WITH attachments
          const adminEncodedMessage = encodeEmailWithAttachments(
            gmailUser,
            `New Part Quotation Request - ${quoteNumber || "New Request"}`,
            adminEmailHtml,
            attachments
          );
          await sendEmail(accessToken, adminEncodedMessage);
          console.log(`Admin notification email sent with ${attachments.length} attachment(s)`);

          // Send customer confirmation email (without attachments)
          const customerEncodedMessage = encodeEmailWithAttachments(
            email,
            `Quotation Request Received - ${quoteNumber || "Your Request"}`,
            customerEmailHtml
          );
          await sendEmail(accessToken, customerEncodedMessage);
          console.log("Customer notification email sent");
        } catch (emailError) {
          console.error("Error sending notification emails:", emailError);
        }
      };

      // Start background email task
      (globalThis as any).EdgeRuntime?.waitUntil(sendNotificationEmails());

      return new Response(
        JSON.stringify({ success: true, message: "Notification emails queued" }),
        { status: 200, headers: { "Content-Type": "application/json", ...corsHeaders } }
      );
    }

    // Extract IP address from request for logging (rate limiting disabled)
    const clientIP =
      req.headers.get("x-forwarded-for")?.split(",")[0].trim() || req.headers.get("x-real-ip") || "unknown";

    const ipHash = await hashIP(clientIP);

    // Validate full quotation input (original flow)
    const validation = validateQuotationInput(rawBody);
    
    if (!validation.valid) {
      console.warn("Input validation failed:", validation.error);
      return new Response(
        JSON.stringify({ error: validation.error }),
        { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
      );
    }
    
    // Sanitize inputs
    const name = sanitizeText(rawBody.name);
    const company = rawBody.company ? sanitizeText(rawBody.company) : undefined;
    const email = rawBody.email.trim().toLowerCase();
    const phone = sanitizeText(rawBody.phone);
    const shippingAddress = rawBody.shippingAddress ? sanitizeText(rawBody.shippingAddress) : '';
    const message = rawBody.message ? sanitizeText(rawBody.message) : undefined;
    const files: FileInfo[] = rawBody.files;
    const drawingFiles: FileInfo[] | undefined = rawBody.drawingFiles;

    console.log("Processing quotation request:", {
      name,
      company,
      email,
      phone,
      filesCount: files.length,
      drawingFilesCount: drawingFiles?.length || 0,
    });

    // Check total attachment size (Gmail limit is 25MB, but we'll be conservative)
    const totalSize =
      files.reduce((sum, f) => sum + f.size, 0) + (drawingFiles?.reduce((sum, f) => sum + f.size, 0) || 0);
    const totalSizeMB = totalSize / 1024 / 1024;

    console.log(`Total attachment size: ${totalSizeMB.toFixed(2)} MB`);

    if (totalSizeMB > 20) {
      console.error("Total attachment size exceeds limit:", totalSizeMB);
      return new Response(
        JSON.stringify({
          error: "file_size_exceeded",
          message: "Total file size exceeds 20MB limit for email attachments",
          totalSizeMB,
        }),
        {
          status: 400,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders,
          },
        },
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
      .from("quotation_submissions")
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
      console.error("Error recording submission:", insertError);
      throw new Error("Failed to create quotation record");
    }

    // Store file metadata in quote_line_items
    const lineItems = [
      ...files.map((file) => ({
        quotation_id: submission.id,
        file_name: file.name,
        file_path: `${submission.id}/cad/${file.name}`,
        quantity: file.quantity,
        material_type: file.material || null,
      })),
      ...(drawingFiles || []).map((file) => ({
        quotation_id: submission.id,
        file_name: file.name,
        file_path: `${submission.id}/drawings/${file.name}`,
        quantity: 1,
      })),
    ];

    let insertedLineItems: any[] = [];
    if (lineItems.length > 0) {
      const { data: inserted, error: lineItemsError } = await supabase
        .from("quote_line_items")
        .insert(lineItems)
        .select();

      if (lineItemsError) {
        console.error("Error storing line items:", lineItemsError);
      } else {
        insertedLineItems = inserted || [];
      }
    }

    // Trigger CAD analysis and preliminary pricing in background
    // Don't await - run asynchronously to avoid blocking response
    if (insertedLineItems.length > 0) {
      Promise.all(
        files.map(async (file, index) => {
          const lineItem = insertedLineItems[index];
          if (!lineItem) return;

          try {
            // Save mesh data and geometric_features to cad_meshes table
            if (file.mesh_data) {
              const fileHash = await hashString(file.name + file.size);

              const { data: meshData, error: meshError } = await supabase
                .from("cad_meshes")
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
                  geometric_features: file.geometric_features || null,
                })
                .select("id")
                .single();

              if (!meshError && meshData) {
                // Update line item with mesh_id
                await supabase.from("quote_line_items").update({ mesh_id: meshData.id }).eq("id", lineItem.id);

                console.log(`✅ Saved mesh data for ${file.name} with geometric_features`);
              } else if (meshError) {
                console.error(`Error saving mesh for ${file.name}:`, meshError);
              }
            }

            // Call analyze-cad function
            const analysisResponse = await fetch(`${supabaseUrl}/functions/v1/analyze-cad`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${supabaseServiceKey}`,
              },
              body: JSON.stringify({
                file_name: file.name,
                file_size: file.size,
                quantity: file.quantity,
              }),
            });

            if (!analysisResponse.ok) {
              console.error(`Analysis failed for ${file.name}`);
              return;
            }

            const analysisData = await analysisResponse.json();

            // Call pricing calculator
            const quoteResponse = await fetch(`${supabaseUrl}/functions/v1/calculate-preliminary-quote`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${supabaseServiceKey}`,
              },
              body: JSON.stringify({
                volume_cm3: analysisData.volume_cm3,
                surface_area_cm2: analysisData.surface_area_cm2,
                complexity_score: analysisData.complexity_score,
                quantity: file.quantity,
                process: "CNC Machining",
                material: "Aluminum 6061",
                finish: "As-machined",
              }),
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
              estimated_machine_time_hours: quoteData.estimated_hours,
            };

            await supabase.from("quote_line_items").update(updateData).eq("id", lineItem.id);

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
                    feature_type: "through_hole",
                    orientation: null,
                    parameters: { ...hole, index: idx },
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
                    feature_type: "blind_hole",
                    orientation: null,
                    parameters: { ...hole, index: idx },
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
                    feature_type: "bore",
                    orientation: null,
                    parameters: { ...bore, index: idx },
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
                    feature_type: "boss",
                    orientation: null,
                    parameters: { ...boss, index: idx },
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
                    feature_type: "planar_face",
                    orientation: null,
                    parameters: { ...face, index: idx },
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
                    feature_type: "fillet",
                    orientation: null,
                    parameters: { ...fillet, index: idx },
                  });
                });
              }

              // Also save the summary for quick access
              if (analysisData.feature_summary) {
                features.push({
                  quotation_id: submission.id,
                  line_item_id: lineItem.id,
                  file_name: file.name,
                  feature_type: "summary",
                  orientation: null,
                  parameters: analysisData.feature_summary,
                });
              }

              if (features.length > 0) {
                const { error: featuresError } = await supabase.from("part_features").insert(features);

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
        }),
      ).catch((err) => console.error("Background analysis error:", err));
    }

    // Send emails in the background to not block the response
    const sendEmails = async () => {
      try {
        const accessToken = await getAccessToken();
        const gmailUser = Deno.env.get("GMAIL_USER") || "belmarj@vectismanufacturing.com";

        // Format shipping address for display
        const formattedAddress = shippingAddress.split("\n").join(", ");

        // Generate admin email content
        const adminDetailsContent = `
          ${generateDetailRow("Quote Number", submission.quote_number)}
          ${generateDetailRow("Date", new Date().toLocaleDateString())}
          ${generateDetailRow("Name", name)}
          ${generateDetailRow("Company", company || "N/A")}
          ${generateDetailRow("Email", email)}
          ${generateDetailRow("Phone", phone || "N/A")}
          ${generateDetailRow("Shipping Address", formattedAddress)}
          ${message ? generateDetailRow("Additional Notes", message) : ""}
        `;

        const adminEmailHtml = generateUnifiedEmailTemplate({
          heroTitle: "New Quote Request",
          heroSubtitle: `A new quotation request has been submitted by <strong>${name}</strong>.<br>Please review the details below.`,
          quoteNumber: submission.quote_number,
          statusStep: 1,
          detailsContent: adminDetailsContent,
          fileListContent: generateFileList(files, drawingFiles),
          timelineText: `Review and respond within <strong>24-48 Hours</strong>`,
          footerText: `${attachments.length} file(s) attached to this email.`,
        });

        // Generate customer email content
        const customerDetailsContent = `
          ${generateDetailRow("Company", company || "N/A")}
          ${generateDetailRow("Total Quantity", `${totalQuantity} Parts`)}
          ${generateDetailRow("Shipping To", formattedAddress)}
        `;

        const customerEmailHtml = generateUnifiedEmailTemplate({
          heroTitle: "Request Received",
          heroSubtitle: `Thanks for your request, <strong>${name}</strong>.<br>Our engineering team has started reviewing your files.`,
          quoteNumber: submission.quote_number,
          statusStep: 1,
          detailsContent: customerDetailsContent,
          fileListContent: generateFileList(files, drawingFiles),
          timelineText: `Estimated Response Time: <strong>24-48 Hours</strong>`,
        });

        // Send admin email with attachments
        const adminEncodedMessage = encodeEmailWithAttachments(
          gmailUser,
          `New Part Quotation Request - ${submission.quote_number}`,
          adminEmailHtml,
          attachments,
        );

        await sendEmail(accessToken, adminEncodedMessage);
        console.log("Admin email sent via Gmail API");

        // Send customer confirmation email (without attachments)
        const customerEncodedMessage = encodeEmailWithAttachments(
          email,
          `Quotation Request Received - ${submission.quote_number}`,
          customerEmailHtml,
        );

        await sendEmail(accessToken, customerEncodedMessage);
        console.log("Customer email sent via Gmail API");
      } catch (emailError) {
        console.error("Error sending emails in background:", emailError);
      }
    };

    // Start background email task (non-blocking)
    (globalThis as any).EdgeRuntime?.waitUntil(sendEmails());

    // Return immediate response without waiting for emails

    return new Response(
      JSON.stringify({
        success: true,
        quoteNumber: submission.quote_number,
      }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders,
        },
      },
    );
  } catch (error: any) {
    console.error("Error in send-quotation-request function:", error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  }
};

serve(handler);
