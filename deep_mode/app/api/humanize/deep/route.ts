import { NextRequest, NextResponse } from "next/server";
import { deepHumanize } from "@/lib/deep_humanizer";
import { ContentCategory } from "@/prompts/deep_prompts";

export const maxDuration = 60;

const VALID_CATEGORIES: ContentCategory[] = ["general", "essay", "academic", "email", "document"];

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { text, category, contentType, wordLimit, temperature, topP } = body as {
      text?: unknown;
      category?: unknown;
      contentType?: unknown;
      wordLimit?: unknown;
      temperature?: unknown;
      topP?: unknown;
    };

    if (typeof text !== "string" || text.trim().length === 0) {
      return NextResponse.json(
        { error: "Text must be a non-empty string." },
        { status: 400 }
      );
    }

    if (text.length > 25000) {
      return NextResponse.json(
        { error: "Input text exceeds maximum allowed length (25,000 characters)." },
        { status: 400 }
      );
    }

    const rawCat = (category || contentType || "general") as string;
    const catLower = rawCat.toLowerCase().trim() as ContentCategory;
    const resolvedCategory: ContentCategory = VALID_CATEGORIES.includes(catLower)
      ? catLower
      : "general";

    const parsedLimit = typeof wordLimit === "number" ? Math.min(Math.max(wordLimit, 50), 2000) : 1000;

    let originalScore = 85;
    let humanizedScore = 15;

    try {
      const { detectAI } = await import("@/lib/detector");
      if (typeof detectAI === "function") {
        originalScore = detectAI(text).score;
      }
    } catch {
    }

    const startTime = Date.now();
    const humanizedText = await deepHumanize(text, resolvedCategory, parsedLimit, {
      temperature: typeof temperature === "number" ? temperature : 0.7,
      topP: typeof topP === "number" ? topP : 0.9,
      wordLimit: parsedLimit,
    });
    const latencyMs = Date.now() - startTime;

    try {
      const { detectAI } = await import("@/lib/detector");
      if (typeof detectAI === "function") {
        humanizedScore = detectAI(humanizedText).score;
      }
    } catch {
    }

    return NextResponse.json({
      humanizedText,
      originalScore,
      humanizedScore,
      category: resolvedCategory,
      mode: "deep",
      model: process.env.LAGOM_DEEP_MODEL_ID || "Aradhya648/lagom-deep-7b",
      latencyMs,
      wordCount: humanizedText.split(/\s+/).filter(Boolean).length,
    });
  } catch (error) {
    console.error("[POST /api/humanize/deep Error]:", error);
    return NextResponse.json(
      {
        error: "Failed to process text in Deep Mode.",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
