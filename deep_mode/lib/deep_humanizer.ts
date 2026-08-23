import { ContentCategory, formatRawChatML } from "../prompts/deep_prompts";

export interface DeepHumanizeOptions {
  temperature?: number;
  topP?: number;
  wordLimit?: number;
  endpointUrl?: string;
  hfToken?: string;
}

export interface DeepHumanizeResponse {
  humanizedText: string;
  category: ContentCategory;
  model: string;
  originalWordCount: number;
  humanizedWordCount: number;
}

const DEFAULT_SERVER_URL = process.env.LAGOM_DEEP_API_URL || "http://localhost:8000";
const HF_MODEL_REPO = process.env.LAGOM_DEEP_MODEL_ID || "Aradhya648/lagom-deep-7b";
const HF_INFERENCE_URL = `https://api-inference.huggingface.co/models/${HF_MODEL_REPO}`;

function splitParagraphChunks(text: string, maxWordsPerChunk: number = 350): string[] {
  const paragraphs = text.split(/\n\s*\n/).map((p) => p.trim()).filter((p) => p.length > 0);
  const chunks: string[] = [];
  let currentChunk: string[] = [];
  let currentWordCount = 0;

  for (const para of paragraphs) {
    const paraWords = para.split(/\s+/).length;
    if (currentWordCount + paraWords > maxWordsPerChunk && currentChunk.length > 0) {
      chunks.push(currentChunk.join("\n\n"));
      currentChunk = [para];
      currentWordCount = paraWords;
    } else {
      currentChunk.push(para);
      currentWordCount += paraWords;
    }
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join("\n\n"));
  }

  return chunks.length > 0 ? chunks : [text.trim()];
}

async function callDeepInference(
  chunk: string,
  category: ContentCategory,
  options: DeepHumanizeOptions = {}
): Promise<string> {
  const serverUrl = options.endpointUrl || process.env.LAGOM_DEEP_API_URL;
  const hfToken = options.hfToken || process.env.HF_TOKEN;

  if (serverUrl) {
    try {
      const resp = await fetch(`${serverUrl}/v1/humanize/deep`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: chunk,
          category: category,
          temperature: options.temperature ?? 0.7,
          top_p: options.topP ?? 0.9,
          word_limit: options.wordLimit ?? 1000,
        }),
      });

      if (resp.ok) {
        const data = await resp.json();
        if (data.humanized_text && data.humanized_text.trim().length > 0) {
          return data.humanized_text.trim();
        }
      }
    } catch (err) {
      console.warn("[Lagom Deep] FastAPI server unreachable, falling back to Hugging Face API:", err);
    }
  }

  const promptStr = formatRawChatML(chunk, category);
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (hfToken) {
    headers["Authorization"] = `Bearer ${hfToken}`;
  }

  try {
    const resp = await fetch(HF_INFERENCE_URL, {
      method: "POST",
      headers,
      body: JSON.stringify({
        inputs: promptStr,
        parameters: {
          max_new_tokens: 1024,
          temperature: options.temperature ?? 0.7,
          top_p: options.topP ?? 0.9,
          return_full_text: false,
        },
      }),
    });

    if (resp.ok) {
      const data = await resp.json();
      if (Array.isArray(data) && data.length > 0 && data[0].generated_text) {
        return data[0].generated_text.replace(/<\|im_end\|>/g, "").trim();
      }
    }
  } catch (err) {
    console.error("[Lagom Deep] HF Inference API error:", err);
  }

  return chunk;
}

export async function deepHumanize(
  text: string,
  category: ContentCategory = "general",
  wordLimit: number = 1000,
  options: DeepHumanizeOptions = {}
): Promise<string> {
  const trimmed = text.trim();
  if (!trimmed) return "";

  const chunks = splitParagraphChunks(trimmed, 350);

  const results = await Promise.all(
    chunks.map((chunk) => callDeepInference(chunk, category, { ...options, wordLimit }))
  );

  let mergedOutput = results.join("\n\n");

  const words = mergedOutput.split(/\s+/);
  if (wordLimit && words.length > wordLimit) {
    mergedOutput = words.slice(0, wordLimit).join(" ");
  }

  return mergedOutput;
}
