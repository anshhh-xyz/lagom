import random
import re
from config import MIN_WORDS, MAX_WORDS, CATEGORY_SIGNALS, REFUSAL_MARKERS


def chunk_text(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        size = random.randint(min_words, max_words)
        chunk = " ".join(words[i:i + size])
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
        i += size
    return chunks


def category_consistency_check(text, category):
    low = text.lower()
    rules = CATEGORY_SIGNALS.get(category, {"include_any": [], "exclude_any": []})

    for bad in rules["exclude_any"]:
        if bad in low:
            return False

    return True


def clean_ai_output(text):
    if not text:
        return ""
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = re.sub(r"(?is)<thought>.*?</thought>", "", text).strip()
    text = re.sub(r"(?i)^\s*(?:\*\*|##)?\s*(?:here(?:'s| is) (?:the |a )?(?:rewritten|revised|formal|ai-style|modified)?\s*(?:text|version|passage|essay|email|report|document)?[:\-]?\s*(?:\*\*|##)?)\s*", "", text)
    text = re.sub(r"(?i)^\s*(?:\*\*|##)?\s*(?:rewritten|revised|aiified)\s*(?:text|version|passage)?[:\-]?\s*(?:\*\*|##)?\s*", "", text)
    text = re.sub(r"(?i)\n+(?:let me know if|i hope this helps|hope this helps|feel free to ask).*$", "", text).strip()
    
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text.strip()


def passes_quality_check(ai_text, human_text):
    if not ai_text or not human_text:
        return False

    ai_clean = clean_ai_output(ai_text)
    if not ai_clean:
        return False

    low = ai_clean.lower()
    if any(m in low[:80] for m in REFUSAL_MARKERS):
        return False

    ai_words = len(ai_clean.split())
    human_words = len(human_text.split())

    if ai_words < MIN_WORDS * 0.6 or ai_words > MAX_WORDS * 1.6:
        return False

    if human_words > 0:
        ratio = ai_words / human_words
        if ratio < 0.55 or ratio > 1.65:
            return False

    if ai_clean.strip() == human_text.strip():
        return False

    return True
