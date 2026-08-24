from config import CATEGORIES


def _fetch_dataset_with_retry(loader_fn, name, retries=3):
    import time
    for attempt in range(1, retries + 1):
        try:
            return loader_fn()
        except Exception as e:
            if attempt < retries:
                print(f"[retry {attempt}/{retries}] Loading {name} failed: {e}. Retrying in 2s...")
                time.sleep(2)
            else:
                print(f"[error] Failed to load {name} after {retries} attempts: {e}")
                return []


def load_human_sources(target_categories=None):
    from datasets import load_dataset

    if target_categories is None:
        target_categories = set(CATEGORIES)
    else:
        target_categories = set(target_categories)

    sources = {c: [] for c in CATEGORIES}

    if "essay" in target_categories:
        def load_essays():
            ds = load_dataset("qwedsacf/ivypanda-essays", split="train")
            return [r["TEXT"] for r in ds if r.get("TEXT")][:6000]
        sources["essay"] = _fetch_dataset_with_retry(load_essays, "ivypanda-essays")

    if "academic" in target_categories:
        def load_academic():
            ds = load_dataset("browndw/human-ai-parallel-corpus", split="train")
            texts = []
            for r in ds:
                doc_id = r.get("doc_id", "")
                if doc_id.startswith("acad_") and ("@chunk" in doc_id or "@" not in doc_id) and r.get("text"):
                    texts.append(r["text"])
                    if len(texts) >= 6000:
                        break
            return texts
        sources["academic"] = _fetch_dataset_with_retry(load_academic, "human-ai-parallel-corpus (academic)")

    if "document" in target_categories:
        def load_docs():
            ds = load_dataset("ccdv/govreport-summarization", split="train")
            return [r["report"] for r in ds if r.get("report")][:5000]
        sources["document"] = _fetch_dataset_with_retry(load_docs, "govreport-summarization")

    if "email" in target_categories:
        def load_emails():
            ds = load_dataset("corbt/enron-emails", split="train")
            emails = []
            for r in ds:
                b = r.get("body")
                if b and len(b.split()) >= 40:
                    emails.append(b)
                    if len(emails) >= 6000:
                        break
            return emails
        sources["email"] = _fetch_dataset_with_retry(load_emails, "corbt/enron-emails")

    if "general" in target_categories:
        def load_blogs():
            ds = load_dataset("tasksource/blog_authorship_corpus", split="train")
            return [r["text"] for r in ds if r.get("text")][:6000]
        sources["general"] = _fetch_dataset_with_retry(load_blogs, "blog_authorship_corpus")

    return sources
