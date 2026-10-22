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
        env_name = KEY_PAIRS[pair_idx]["gemini_env"] if pair_idx < len(KEY_PAIRS) else f"GEMINI_API_KEY_{pair_idx+1}"
        candidates = [
            env_name,
            f"GEMINI_API_KEY_{pair_idx+1}",
            f"GEMINI_API_KEY{pair_idx+1}",
            f"GEMINI_API_KEY_{pair_idx}",
            "GEMINI_API_KEY",
            "GEMINI_API_KEY_1",
        ]
        key = get_api_key(*candidates)
        if not key:
            raise KeyError(f"No Gemini API key found for slot {pair_idx} (checked {candidates}).")
        _gemini_clients[pair_idx] = genai.Client(api_key=key)
    return _gemini_clients[pair_idx]


def get_groq_client(pair_idx=0):
    if pair_idx not in _groq_clients:
        from groq import Groq
        env_name = KEY_PAIRS[pair_idx]["groq_env"] if pair_idx < len(KEY_PAIRS) else f"GROQ_API_KEY_{pair_idx+1}"
        candidates = [
            env_name,
            f"GROQ_API_KEY_{pair_idx+1}",
            f"GROQ_API_KEY{pair_idx+1}",
            f"GROQ_API_KEY_{pair_idx}",
            "GROQ_API_KEY",
            "GROQ_API_KEY_1",
        ]
        key = get_api_key(*candidates)
        if not key:
            raise KeyError(f"No Groq API key found for slot {pair_idx} (checked {candidates}).")
        _groq_clients[pair_idx] = Groq(api_key=key)
    return _groq_clients[pair_idx]
