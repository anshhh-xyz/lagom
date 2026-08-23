OUTPUT_CSV = "lagom_pairs.csv"
MISMATCH_CSV = "lagom_pairs_flagged.csv"
MIN_WORDS = 50
MAX_WORDS = 400
TARGET_ROWS_PER_CATEGORY = 6000
RATE_LIMIT_SLEEP = 0.02

CATEGORIES = ["general", "essay", "academic", "email", "document"]

KEY_PAIRS = [
    {"name": "Pair 1", "gemini_env": "GEMINI_API_KEY",  "groq_env": "GROQ_API_KEY"},
    {"name": "Pair 2", "gemini_env": "GEMINI_API_KEY2", "groq_env": "GROQ_API_KEY2"},
    {"name": "Pair 3", "gemini_env": "GEMINI_API_KEY3", "groq_env": "GROQ_API_KEY3"},
    {"name": "Pair 4", "gemini_env": "GEMINI_API_KEY4", "groq_env": "GROQ_API_KEY4"},
    {"name": "Pair 5", "gemini_env": "GEMINI_API_KEY5", "groq_env": "GROQ_API_KEY5"},
]

AIIFY_PROMPTS = {
    "general": (
        "Rewrite the following passage the way a generic AI writing assistant "
        "would: add slight formal tone, generic transition phrases "
        "(e.g. 'Moreover', 'In addition', 'It is important to note'), and mild "
        "over-explanation. Keep the same meaning and length roughly the same. "
        "Return ONLY the rewritten text, no preamble.Make it as a general chat done by AI\n\nTEXT:\n{text}"
    ),
    "essay": (
        "Rewrite the following passage in the style of a typical AI-generated "
        "essay: structured, slightly repetitive sentence rhythm, heavy use of "
        "'furthermore/moreover/in conclusion' style transitions, and a "
        "tendency to summarize points that were already made. Return ONLY the "
        "rewritten text, no preamble.Make it as a general essay would be written by AI\n\nTEXT:\n{text}"
    ),
    "academic": (
        "Rewrite the following passage in the style of AI-generated academic "
        "writing: overly hedged claims ('it could be argued that', 'may "
        "suggest'), excessive formality, and generic academic transitions. "
        "Do not fabricate citations. Return ONLY the rewritten text, no "
        "preamble.\n\nTEXT:\n{text}"
    ),
    "email": (
        "Rewrite the following passage as an overly formal, slightly robotic "
        "AI-assistant-style email: stiff greeting/sign-off structure, "
        "over-polite hedging, and generic corporate phrasing. Return ONLY the "
        "rewritten text, no preamble.An email written by AI\n\nTEXT:\n{text}"
    ),
    "document": (
        "Rewrite the following passage in the style of an AI-generated formal "
        "document/report: heavy structuring, bullet-point tendencies even in "
        "prose form, generic business/report language. Return ONLY the "
        "rewritten text, no preamble.A document made by AI\n\nTEXT:\n{text}"
    ),
}

CATEGORY_SIGNALS = {
    "email": {
        "include_any": [
            "dear ", "hi ", "hello ", "regards", "sincerely", "best,",
            "subject:", "thanks,", "thank you,", "cc:", "forwarded message",
        ],
        "exclude_any": [],
    },
    "essay": {
        "include_any": [],
        "exclude_any": [
            "dear sir", "dear madam", "subject:", "sincerely,", "regards,",
        ],
    },
    "academic": {
        "include_any": [
            "abstract", "et al", "this study", "results show", "hypothesis",
            "methodology", "we propose", "findings suggest",
        ],
        "exclude_any": ["dear ", "subject:", "sincerely,"],
    },
    "document": {
        "include_any": [],
        "exclude_any": ["dear sir", "dear madam", "love,", "xoxo"],
    },
    "general": {
        "include_any": [],
        "exclude_any": [],
    },
}

REFUSAL_MARKERS = [
    "i cannot", "i can't", "as an ai", "here is the rewritten",
    "here's the rewritten", "sure, here", "i'm sorry",
]
