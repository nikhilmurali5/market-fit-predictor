/**
 * AI Recommendation Module (GROQ ONLY - CLEAN TEST)
 */

const axios = require("axios");

// ─── Build prompt ─────────────────────────────────────────
function buildPrompt(category, specifications, score) {
  const specLines = Object.entries(specifications)
    .map(([k, v]) => `${k}: ${v}`)
    .join(", ");

  return `
You are a professional product analyst.

Analyze the product based on its specifications and give smart recommendations.

Category: ${category}
Market Score: ${score}/100
Specs: ${specLines}

Instructions:
- Give ONLY 5 bullet points
- Each point must be specific and realistic
- If a spec is LOW → suggest improvement
- If a spec is GOOD → highlight it as a strength
- If a spec is AVERAGE → give a balanced opinion
- Do NOT suggest improvements for everything
- Avoid repeating the same type of suggestion

Focus on:
• performance
• battery
• camera
• usability
• target users

Example of GOOD output:
- Good for daily use and multitasking with 8GB RAM
- Storage is excellent and more than enough for most users
- Battery is decent but may need improvement for heavy users
- Camera performs well in daylight but average in low light
- Suitable for students and casual users, not ideal for gaming

Now generate the recommendations:
`;
}

// ─── GROQ ONLY ────────────────────────────────────────────
async function callGroq(prompt) {
  try {
    const res = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        model: "llama-3.1-8b-instant",
        messages: [
          { role: "user", content: prompt }
        ],
        temperature: 0.7
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          "Content-Type": "application/json"
        }
      }
    );

    console.log("FULL GROQ RESPONSE:", JSON.stringify(res.data, null, 2));

    const text = res.data?.choices?.[0]?.message?.content || "";

    console.log("⚡ GROQ TEXT:", text);

    return text;

  } catch (err) {
    console.log("❌ GROQ ERROR:", err.response?.data || err.message);
    return "";
  }
}

// ─── MAIN FUNCTION ────────────────────────────────────────
async function getAIRecommendation(category, specifications, score) {
  const prompt = buildPrompt(category, specifications, score);

  let raw = "";

  try {
    raw = await callGroq(prompt);   // ✅ ONLY GROQ
  } catch (err) {
    console.error("❌ GROQ FAILED:", err.message);
    raw = "";
  }

  console.log("🧠 FINAL AI OUTPUT:", raw);

  return {
    recommendation: raw || "AI could not generate recommendation"
  };
}

module.exports = { getAIRecommendation };