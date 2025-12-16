import { serve } from "https://deno.land/std@0.190.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface ConfirmationRequest {
  customerName: string;
  customerEmail: string;
  quoteNumber: string;
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
  const gmailUser = Deno.env.get("GMAIL_USER") || "info@vectismanufacturing.com";
  
  const emailLines = [
    `From: Vectis Manufacturing <${gmailUser}>`,
    `To: ${to}`,
    `Subject: ${subject}`,
    "MIME-Version: 1.0",
    'Content-Type: text/html; charset="UTF-8"',
    "",
    htmlBody,
  ];
  
  const email = emailLines.join("\r\n");
  return btoa(unescape(encodeURIComponent(email)))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

function generateConfirmationEmail(customerName: string, quoteNumber: string, partDetails?: ConfirmationRequest['partDetails']): string {
  const partDetailsSection = partDetails ? `
    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(99, 102, 241, 0.05)); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 20px; margin: 20px 0;">
      <h3 style="color: #1e40af; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 15px 0; padding-bottom: 10px; border-bottom: 1px solid rgba(59, 130, 246, 0.2);">
        📋 Your Request Details
      </h3>
      <table style="width: 100%; border-collapse: collapse;">
        ${partDetails.partName ? `<tr><td style="padding: 8px 0; color: #64748b; font-size: 13px; width: 40%;">Part Name:</td><td style="padding: 8px 0; color: #1e293b; font-size: 13px; font-weight: 500;">${partDetails.partName}</td></tr>` : ''}
        ${partDetails.material ? `<tr><td style="padding: 8px 0; color: #64748b; font-size: 13px;">Material:</td><td style="padding: 8px 0; color: #1e293b; font-size: 13px; font-weight: 500;">${partDetails.material}</td></tr>` : ''}
        ${partDetails.quantity ? `<tr><td style="padding: 8px 0; color: #64748b; font-size: 13px;">Quantity:</td><td style="padding: 8px 0; color: #1e293b; font-size: 13px; font-weight: 500;">${partDetails.quantity}</td></tr>` : ''}
        ${partDetails.finish ? `<tr><td style="padding: 8px 0; color: #64748b; font-size: 13px;">Finish:</td><td style="padding: 8px 0; color: #1e293b; font-size: 13px; font-weight: 500;">${partDetails.finish}</td></tr>` : ''}
        ${partDetails.heatTreatment ? `<tr><td style="padding: 8px 0; color: #64748b; font-size: 13px;">Heat Treatment:</td><td style="padding: 8px 0; color: #1e293b; font-size: 13px; font-weight: 500;">Required${partDetails.heatTreatmentDetails ? ` - ${partDetails.heatTreatmentDetails}` : ''}</td></tr>` : ''}
        ${partDetails.threadsTolerances ? `<tr><td style="padding: 8px 0; color: #64748b; font-size: 13px;">Threads/Tolerances:</td><td style="padding: 8px 0; color: #1e293b; font-size: 13px; font-weight: 500;">${partDetails.threadsTolerances}</td></tr>` : ''}
      </table>
    </div>
  ` : '';

  return `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f1f5f9;">
      <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border-radius: 12px 12px 0 0; padding: 30px; text-align: center;">
          <h1 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: 600;">Quote Request Received</h1>
          <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0; font-size: 14px;">Reference: ${quoteNumber}</p>
        </div>
        
        <!-- Body -->
        <div style="background-color: #ffffff; padding: 30px; border-radius: 0 0 12px 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
          <p style="color: #334155; font-size: 16px; line-height: 1.6; margin: 0 0 20px 0;">
            Dear ${customerName},
          </p>
          
          <p style="color: #334155; font-size: 15px; line-height: 1.6; margin: 0 0 20px 0;">
            Thank you for your quote request! We have successfully received your submission and our team is reviewing your requirements.
          </p>
          
          <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 15px; margin: 20px 0;">
            <p style="color: #166534; font-size: 14px; margin: 0; display: flex; align-items: center;">
              <span style="font-size: 18px; margin-right: 10px;">✅</span>
              <span><strong>Your reference number:</strong> ${quoteNumber}</span>
            </p>
          </div>

          ${partDetailsSection}
          
          <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 15px; margin: 20px 0;">
            <p style="color: #1e40af; font-size: 14px; margin: 0;">
              <strong>What happens next?</strong><br><br>
              Our engineering team will review your CAD files and specifications. You can expect to receive a detailed quote within <strong>1-2 business days</strong>.
            </p>
          </div>
          
          <p style="color: #64748b; font-size: 14px; line-height: 1.6; margin: 20px 0 0 0;">
            If you have any questions in the meantime, please don't hesitate to contact us at <a href="mailto:info@vectismanufacturing.com" style="color: #2563eb;">info@vectismanufacturing.com</a>.
          </p>
          
          <p style="color: #334155; font-size: 14px; margin: 30px 0 0 0;">
            Best regards,<br>
            <strong>The Vectis Manufacturing Team</strong>
          </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 20px; color: #94a3b8; font-size: 12px;">
          <p style="margin: 0;">© ${new Date().getFullYear()} Vectis Manufacturing. All rights reserved.</p>
          <p style="margin: 5px 0 0 0;">Precision CNC Machining & Custom Manufacturing</p>
        </div>
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
    const { customerName, customerEmail, quoteNumber, partDetails }: ConfirmationRequest = await req.json();

    console.log("Sending quote request confirmation to:", customerEmail);

    // Generate email content
    const htmlBody = generateConfirmationEmail(customerName, quoteNumber, partDetails);
    
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
