import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { features, material, volume_cm3, part_name } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");

    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY not configured");
    }

    // Build technical summary from features
    const featureSummary = features.map((f: any) => {
      const dims = f.dimensions || {};
      let desc = `${f.type}`;
      
      if (f.subtype) desc += ` (${f.subtype})`;
      if (dims.diameter) desc += ` - Ø${dims.diameter.toFixed(1)}mm`;
      if (dims.depth) desc += `, depth ${dims.depth.toFixed(1)}mm`;
      if (dims.width && dims.length) {
        desc += ` - ${dims.width.toFixed(1)}mm × ${dims.length.toFixed(1)}mm`;
      }
      if (dims.radius) desc += ` - R${dims.radius.toFixed(1)}mm`;
      
      return desc;
    }).join('\n');

    // Construct AI prompt
    const prompt = `You are a manufacturing engineer explaining CAD analysis results to a customer.

**Part Details**:
- Name: ${part_name}
- Material: ${material || 'Not specified'}
- Volume: ${volume_cm3} cm³

**Detected Manufacturing Features**:
${featureSummary}

Generate a customer-friendly analysis with:

1. **Summary** (2-3 sentences): Brief description of what this part is and its key characteristics
2. **Required Machining Operations** (bullet list): What manufacturing processes will be needed
3. **Complexity Assessment**: Rate as Simple, Medium, or Complex with brief explanation
4. **Design Recommendations** (if applicable): Any tips for easier/cheaper manufacturing

Keep it professional, concise, and avoid excessive jargon. Focus on practical manufacturing insights.`;

    // Call Lovable AI Gateway
    const aiResponse = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { 
            role: "system", 
            content: "You are an expert manufacturing engineer who explains CAD analysis in clear, customer-friendly language." 
          },
          { role: "user", content: prompt }
        ],
      }),
    });

    if (!aiResponse.ok) {
      const errorText = await aiResponse.text();
      console.error("AI gateway error:", aiResponse.status, errorText);
      
      if (aiResponse.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded. Please try again in a moment." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      
      if (aiResponse.status === 402) {
        return new Response(
          JSON.stringify({ error: "AI credits exhausted. Please add credits to your workspace." }),
          { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      
      throw new Error(`AI gateway failed: ${aiResponse.status}`);
    }

    const aiData = await aiResponse.json();
    const explanation = aiData.choices[0].message.content;

    return new Response(
      JSON.stringify({ 
        explanation,
        model_used: "google/gemini-2.5-flash",
        timestamp: new Date().toISOString()
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (error) {
    console.error("Feature explanation error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
