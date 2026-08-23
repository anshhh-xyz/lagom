export type ContentCategory = "general" | "essay" | "academic" | "email" | "document";

export const LAGOM_DEEP_SYSTEM_PROMPT =
  "You are Lagom, a specialized humanizer AI. Your task is to rewrite the provided " +
  "AI-generated text so it sounds naturally human-authored, organic, and nuanced while " +
  "preserving 100% of the original factual meaning and core ideas. Vary sentence lengths, " +
  "use natural syntactic rhythms, eliminate robotic transition cliches, and match the " +
  "target genre tone.";

export const DEEP_CATEGORY_INSTRUCTIONS: Record<ContentCategory, string> = {
  general:
    "Rewrite this text into natural, organic human writing. Remove artificial AI cadence, " +
    "repetitive transitions, and generic over-explanations while keeping the meaning intact.",
  essay:
    "Rewrite this essay passage in the voice of a skilled human writer. Vary the sentence " +
    "rhythm naturally, remove formulaic signposts (e.g. 'furthermore', 'moreover', 'in conclusion'), " +
    "and preserve genuine thesis flow without repetitive summary statements.",
  academic:
    "Rewrite this academic passage in genuine, human-authored scholarly prose. Use precise, " +
    "substantive vocabulary without artificial buzzwords or robotic hedging patterns.",
  email:
    "Rewrite this email to sound like an authentic human professional. Use warm, natural " +
    "greetings and sign-offs, realistic phrasing, and eliminate stiff corporate AI cliches.",
  document:
    "Rewrite this document/report in clean, natural human report style. Ensure sharp readability, " +
    "logical structural hierarchy, and authentic business prose.",
};

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export function formatDeepChatMessages(text: string, category: ContentCategory = "general"): ChatMessage[] {
  const instruction = DEEP_CATEGORY_INSTRUCTIONS[category] || DEEP_CATEGORY_INSTRUCTIONS.general;
  const userContent = `${instruction}\n\n[TEXT TO HUMANIZE]:\n${text.trim()}`;

  return [
    { role: "system", content: LAGOM_DEEP_SYSTEM_PROMPT },
    { role: "user", content: userContent },
  ];
}

export function formatRawChatML(text: string, category: ContentCategory = "general"): string {
  const instruction = DEEP_CATEGORY_INSTRUCTIONS[category] || DEEP_CATEGORY_INSTRUCTIONS.general;
  return (
    `<|im_start|>system\n${LAGOM_DEEP_SYSTEM_PROMPT}<|im_end|>\n` +
    `<|im_start|>user\n${instruction}\n\n[TEXT TO HUMANIZE]:\n${text.trim()}<|im_end|>\n` +
    `<|im_start|>assistant\n`
  );
}
