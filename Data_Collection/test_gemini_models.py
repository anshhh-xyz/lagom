import requests
from config import KEY_PAIRS
from clients import get_api_key

gemini_models_to_test = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-pro",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it"
]

for pair in KEY_PAIRS:
    name = pair["name"]
    env = pair["gemini_env"]
    key = get_api_key(env)
    print(f"\n[{name}] Gemini Key ({env})")
    for m in gemini_models_to_test:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
        payload = {
            "contents": [{"parts": [{"text": "hi"}]}],
            "generationConfig": {"maxOutputTokens": 1}
        }
        try:
            r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=8)
            if r.status_code == 200:
                print(f"  * {m}: ACTIVE (200 OK)")
            elif r.status_code == 429:
                err = r.json().get("error", {}).get("message", "")
                print(f"  * {m}: 429 QUOTA EXHAUSTED -> {err[:120]}...")
            elif r.status_code == 404:
                err = r.json().get("error", {}).get("message", "")
                print(f"  * {m}: 404 NOT FOUND -> {err[:80]}...")
            else:
                print(f"  * {m}: HTTP {r.status_code} -> {r.text[:80]}...")
        except Exception as e:
            print(f"  * {m}: ERROR -> {e}")
