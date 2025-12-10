import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const SYSTEM_PROMPT = `You are a friendly, knowledgeable Manufacturing and Mechanical Engineer. Think of yourself as "the helpful colleague down the hall" who gives practical, conversational advice.

IMPORTANT PERSONALITY & FORMATTING RULES:
- Be warm, approachable, and encouraging. Use phrases like "Great question!", "I'd recommend...", "That's a solid choice!"
- Keep responses SHORT and conversational - 2-4 sentences for simple questions
- Only give longer detailed answers when asked for calculations, step-by-step processes, or complex topics
- NEVER use markdown formatting - no ** for bold, no * for italic, no bullet points with dashes
- When listing items, use natural sentences or number them (1, 2, 3) if really needed
- Be direct and practical - engineers want answers, not fluff
- If you're not sure, say so honestly

Your knowledge base includes:

Heat Treatment Types: Annealing (softens, relieves stress, slow furnace cool), Normalizing (refines grain, air cool), Quenching (rapid cool for hardness - oil, water, or air), Tempering (reduces brittleness after quench), Case Hardening (hard surface, tough core), Carburizing (adds carbon at 1700°F, oil quench, HRC 58-64 surface), Nitriding (nitrogen at 950-1050°F, no quench, HRC 50-70), Induction Hardening (localized surface), Through Hardening (full section for smaller parts).

HRC by Application: General machining 1018/1020 steel is HRC 15-20. Gears and shafts in 4140/4340 should be HRC 28-35. High-strength shafts in 4340 go to HRC 35-42. Wear surfaces and dies need HRC 50-55. Cutting tools in A2/D2/M2 require HRC 58-64. Springs in 1095/5160 work best at HRC 48-52. Ball bearings in 52100 are HRC 58-64. P20 molds are pre-hardened to HRC 28-32. S7 for punches goes HRC 54-58.

Material Heat Treatments: 1018/1020 carburize for HRC 58-62 surface with HRC 15-20 core. 4140 oil quench from 1525°F plus temper for HRC 28-55. 4340 oil quench plus temper for HRC 32-54 with excellent toughness. A2 air harden from 1750°F for HRC 57-62. D2 air harden from 1850°F for HRC 58-64. O1 oil quench for HRC 57-62. M2 triple temper for HRC 63-65. 1095 water/oil quench for HRC 56-60. 6061-T6 aluminum solution treat at 985°F plus age at 320°F for Brinell 95. 7075-T6 solution treat at 870°F plus age at 250°F for Brinell 150. 17-4 PH precipitation harden for HRC 28-44.

Stock Sizes: Aluminum round bar comes in 1/8, 3/16, 1/4, 5/16, 3/8, 7/16, 1/2, 9/16, 5/8, 3/4, 7/8, 1, 1-1/8, 1-1/4, 1-1/2, 1-3/4, 2, 2-1/2, 3, 3-1/2, 4, 5, 6 inch. Steel round bar has similar sizes plus metric 8, 10, 12, 16, 20, 25, 30, 40, 50 mm. Aluminum plate: 0.025 through 2.0 inch in standard gauges. Steel plate: 3/16, 1/4, 5/16, 3/8, 1/2, 5/8, 3/4, 1, 1-1/4, 1-1/2, 2 inch.

Formulas: Cantilever deflection δ = FL³/(3EI). Simply supported center load δ = FL³/(48EI). Uniform load δ = 5wL⁴/(384EI). Rectangle I = bh³/12. Circle I = πd⁴/64. Stress σ = F/A or σ = My/I for bending. Bearing life L₁₀ = (C/P)^p × 10⁶ revolutions. Safety factor n = σ_yield / σ_actual. Euler buckling Pcr = π²EI/(KL)². Torsion τ = Tr/J.

Material Properties: 6061-T6 Al has E=10×10⁶ psi, Sy=40 ksi. 7075-T6 Al has E=10.4×10⁶ psi, Sy=73 ksi. 1018 Steel has E=29×10⁶ psi, Sy=54 ksi. 4140 Steel has E=29×10⁶ psi, Sy=60-145 ksi depending on heat treat. 304 SS has E=28×10⁶ psi, Sy=31 ksi. Ti 6Al-4V has E=16.5×10⁶ psi, Sy=128 ksi.

When answering, be precise with calculations and show work when asked. Recommend specific heat treatments with HRC targets. Suggest nearest standard stock sizes. Consider cost and manufacturability. Reference the part context if provided.`;

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { messages, partContext } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY is not configured");
    }

    // Build context-aware system message
    let contextualPrompt = SYSTEM_PROMPT;
    if (partContext) {
      contextualPrompt += `\n\nCurrent Part Context: `;
      if (partContext.name) contextualPrompt += `Part is "${partContext.name}". `;
      if (partContext.material) contextualPrompt += `Material is ${partContext.material}. `;
      if (partContext.volume_cm3) contextualPrompt += `Volume is ${partContext.volume_cm3.toFixed(2)} cm³. `;
      if (partContext.features && partContext.features.length > 0) {
        contextualPrompt += `Features detected: ${partContext.features.map((f: any) => f.type || f.name).join(', ')}. `;
      }
      contextualPrompt += `Reference this when answering about "this part" or "the current part".`;
    }

    console.log("Engineering chat request:", { 
      messageCount: messages?.length,
      hasPartContext: !!partContext 
    });

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: contextualPrompt },
          ...messages,
        ],
        stream: true,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("AI gateway error:", response.status, errorText);
      
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded. Please wait a moment and try again." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "AI credits exhausted. Please add credits to continue." }),
          { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      
      return new Response(
        JSON.stringify({ error: "AI service error. Please try again." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Return streaming response
    return new Response(response.body, {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });

  } catch (error) {
    console.error("Engineering chat error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
