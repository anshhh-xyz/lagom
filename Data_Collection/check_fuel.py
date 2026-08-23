import requests
from config import KEY_PAIRS
from clients import get_api_key

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
]

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
]


def check_groq_pair(key):
    if not key:
        return {"status": "MISSING", "models": {}}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    model_stats = {}
    
    for m in GROQ_MODELS:
        payload = {
            "model": m,
            "messages": [{"role": "user", "content": "1"}],
            "max_tokens": 1
        }
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=8
            )
            h = resp.headers
            if resp.status_code == 200:
                model_stats[m] = {
                    "status": "ACTIVE",
                    "req_remaining": h.get("x-ratelimit-remaining-requests"),
                    "req_limit": h.get("x-ratelimit-limit-requests"),
                    "tok_remaining": h.get("x-ratelimit-remaining-tokens"),
                    "tok_limit": h.get("x-ratelimit-limit-tokens"),
                    "reset_req": h.get("x-ratelimit-reset-requests"),
                    "reset_tok": h.get("x-ratelimit-reset-tokens"),
                }
            elif resp.status_code == 429:
                model_stats[m] = {"status": "429 RATE LIMITED / EXHAUSTED"}
            else:
                model_stats[m] = {"status": f"HTTP {resp.status_code}"}
        except Exception as e:
            model_stats[m] = {"status": f"ERROR: {e}"}
            
    return {"status": "OK", "models": model_stats}


def check_gemini_pair(key):
    if not key:
        return {"status": "MISSING", "models": {}}
    model_stats = {}
    
    for m in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
        payload = {
            "contents": [{"parts": [{"text": "hi"}]}],
            "generationConfig": {"maxOutputTokens": 1}
        }
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=8
            )
            if resp.status_code == 200:
                model_stats[m] = {"status": "ACTIVE (Ready)"}
            elif resp.status_code == 429:
                err_msg = resp.json().get("error", {}).get("message", "Quota Exceeded")
                model_stats[m] = {"status": "429 QUOTA EXHAUSTED", "detail": err_msg[:120]}
            elif resp.status_code == 404:
                model_stats[m] = {"status": "404 NOT FOUND"}
            else:
                model_stats[m] = {"status": f"HTTP {resp.status_code}"}
        except Exception as e:
            model_stats[m] = {"status": f"ERROR: {e}"}
            
    return {"status": "OK", "models": model_stats}


def main():
    print("=" * 85)
    print("                LAGOM PIPELINE - API KEY PAIRS FUEL INSPECTION")
    print("=" * 85)
    
    for pair in KEY_PAIRS:
        name = pair["name"]
        g_env = pair["gemini_env"]
        gr_env = pair["groq_env"]
        
        g_key = get_api_key(g_env)
        gr_key = get_api_key(gr_env)
        
        print(f"\n[{name}]")
        print("-" * 85)
        
        print(f" Groq ({gr_env} | {'...' + gr_key[-6:] if gr_key else 'NOT FOUND'}):")
        groq_res = check_groq_pair(gr_key)
        for m, stats in groq_res["models"].items():
            if stats.get("status") == "ACTIVE":
                print(f"   * {m:22}: [ACTIVE] Requests: {stats['req_remaining']}/{stats['req_limit']} (Resets in: {stats['reset_req']}) | Tokens: {stats['tok_remaining']}/{stats['tok_limit']}")
            else:
                print(f"   * {m:22}: [{stats.get('status')}]")
                
        print(f" Gemini ({g_env} | {'...' + g_key[-6:] if g_key else 'NOT FOUND'}):")
        gem_res = check_gemini_pair(g_key)
        for m, stats in gem_res["models"].items():
            status = stats.get("status")
            detail = f" -> {stats.get('detail')}" if "detail" in stats else ""
            print(f"   * {m:22}: [{status}]{detail}")

    print("\n" + "=" * 85)


if __name__ == "__main__":
    main()
