import os
from config import KEY_PAIRS

_gemini_clients = {}
_groq_clients = {}
_openai_client = None
_grok_client = None


def get_api_key(primary_name, *fallback_names):
    names = (primary_name, *fallback_names)
    for n in names:
        if n in os.environ and os.environ[n]:
            return os.environ[n].strip()
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                for n in names:
                    try:
                        val, _ = winreg.QueryValueEx(key, n)
                        if val:
                            return val.strip()
                    except FileNotFoundError:
                        pass
        except Exception:
            pass
    return None


def get_gemini_client(pair_idx=0):
    if pair_idx not in _gemini_clients:
        from google import genai
        env_name = KEY_PAIRS[pair_idx]["gemini_env"]
        key = get_api_key(env_name)
        if not key:
            raise KeyError(f"{env_name} environment variable not found for {KEY_PAIRS[pair_idx]['name']}.")
        _gemini_clients[pair_idx] = genai.Client(api_key=key)
    return _gemini_clients[pair_idx]


def get_groq_client(pair_idx=0):
    if pair_idx not in _groq_clients:
        from openai import OpenAI
        env_name = KEY_PAIRS[pair_idx]["groq_env"]
        key = get_api_key(env_name)
        if not key:
            raise KeyError(f"{env_name} environment variable not found for {KEY_PAIRS[pair_idx]['name']}.")
        _groq_clients[pair_idx] = OpenAI(
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq_clients[pair_idx]
