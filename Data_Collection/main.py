import csv
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from config import (
    OUTPUT_CSV,
    MISMATCH_CSV,
    TARGET_ROWS_PER_CATEGORY,
    RATE_LIMIT_SLEEP,
    CATEGORIES,
    KEY_PAIRS,
)
from sources import load_human_sources
from text_utils import (
    chunk_text,
    category_consistency_check,
    passes_quality_check,
    clean_ai_output,
)
from generator import aiify_call


def load_already_done_stats(path):
    done_texts = set()
    category_counts = {c: 0 for c in CATEGORIES}
    if not Path(path).exists():
        return done_texts, category_counts
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                done_texts.add(row[1].strip())
                cat = row[2].strip()
                category_counts[cat] = category_counts.get(cat, 0) + 1
    return done_texts, category_counts


def run_pipeline():
    file_exists = Path(OUTPUT_CSV).exists()
    csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow([
            "AI Text", "Human Text", "category", "source_model",
            "word_count", "human_source", "contributor", "reviewed",
        ])
        csv_file.flush()

    mismatch_exists = Path(MISMATCH_CSV).exists()
    mismatch_file = open(MISMATCH_CSV, "a", newline="", encoding="utf-8")
    mwriter = csv.writer(mismatch_file)
    if not mismatch_exists:
        mwriter.writerow(["Human Text (unpaired)", "assigned_category", "reason"])
        mismatch_file.flush()

    csv_lock = threading.Lock()
    mismatch_lock = threading.Lock()

    already_done, category_counts = load_already_done_stats(OUTPUT_CSV)
    print("=" * 80, flush=True)
    print(f"Resuming Pipeline: {len(already_done)} unique rows in {OUTPUT_CSV}.", flush=True)
    for cat, cnt in category_counts.items():
        status = "[DONE - Target Reached]" if cnt >= TARGET_ROWS_PER_CATEGORY else f"{cnt}/{TARGET_ROWS_PER_CATEGORY} rows"
        print(f" - {cat:12}: {status}", flush=True)
    print("=" * 80, flush=True)

    print("\nLoading human source datasets...", flush=True)
    sources = load_human_sources()

    num_workers = len(KEY_PAIRS) * 2
    print(f"\n[Parallel Engine] Starting {num_workers} Concurrent Workers (2 per Key Pair across 5 pairs):", flush=True)
    for i, p in enumerate(KEY_PAIRS):
        print(f"  * Workers {i*2+1}, {i*2+2} -> {p['name']} ({p['gemini_env']} + {p['groq_env']})", flush=True)

    stop_event = threading.Event()

    for category in CATEGORIES:
        completed_in_cat = category_counts.get(category, 0)
        needed = TARGET_ROWS_PER_CATEGORY - completed_in_cat
        if needed <= 0:
            print(f"\n=== {category.upper()}: TARGET REACHED ({completed_in_cat}/{TARGET_ROWS_PER_CATEGORY}) -> SKIPPING ===", flush=True)
            continue

        raw_texts = sources.get(category, [])
        if not raw_texts:
            print(f"[skip] no source text for category={category}", flush=True)
            continue

        random.seed(42)
        chunks = []
        for t in raw_texts:
            chunks.extend(chunk_text(t))
        random.shuffle(chunks)

        clean_chunks = []
        for c in chunks:
            if category_consistency_check(c, category):
                clean_chunks.append(c)
            else:
                with mismatch_lock:
                    mwriter.writerow([c[:500], category, "failed exclude_any heuristic"])
                    mismatch_file.flush()

        remaining_chunks = [c for c in clean_chunks if c.strip() not in already_done]
        queue_chunks = remaining_chunks[: int(needed * 1.3) + 100]

        print(
            f"\n=== {category.upper()}: {len(queue_chunks)} chunks queued to reach {needed} needed "
            f"({completed_in_cat} already completed, Target: {TARGET_ROWS_PER_CATEGORY}) ===",
            flush=True
        )

        chunk_queue = queue.Queue()
        for c in queue_chunks:
            chunk_queue.put(c)

        pbar = tqdm(total=needed, initial=0, desc=f"{category[:10]:10}")
        current_cat_completed = 0
        cat_lock = threading.Lock()

        def worker_loop(pair_idx):
            nonlocal current_cat_completed
            model_cycle = ["groq", "gemini"]
            model_i = pair_idx

            while not stop_event.is_set():
                with cat_lock:
                    if current_cat_completed >= needed:
                        break
                try:
                    chunk = chunk_queue.get(timeout=1.0)
                except queue.Empty:
                    break

                preferred_model = model_cycle[model_i % len(model_cycle)]
                ai_text = None
                used_model_name = None

                try:
                    ai_text, used_model_name = aiify_call(
                        chunk, category, model=preferred_model, pair_idx=pair_idx
                    )
                    model_i += 1
                except Exception:
                    time.sleep(3.0)
                    chunk_queue.put(chunk)
                    continue

                if not ai_text or not passes_quality_check(ai_text, chunk):
                    continue

                ai_final = clean_ai_output(ai_text)

                with csv_lock:
                    if chunk.strip() in already_done:
                        continue
                    writer.writerow([
                        ai_final,
                        chunk,
                        category,
                        used_model_name or preferred_model,
                        len(chunk.split()),
                        "hf_dataset_import",
                        "script",
                        False,
                    ])
                    csv_file.flush()
                    already_done.add(chunk.strip())

                with cat_lock:
                    current_cat_completed += 1
                    pbar.update(1)

                time.sleep(RATE_LIMIT_SLEEP)

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_loop, i % len(KEY_PAIRS)) for i in range(num_workers)]
                for f in futures:
                    f.result()
        except (KeyboardInterrupt, SystemExit):
            print("\n[!] User interrupted pipeline. Exiting cleanly...", flush=True)
            stop_event.set()
            try:
                csv_file.flush()
                csv_file.close()
                mismatch_file.flush()
                mismatch_file.close()
            except Exception:
                pass
            os._exit(0)

        pbar.close()
        category_counts[category] = completed_in_cat + current_cat_completed

    csv_file.close()
    mismatch_file.close()
    total_done_final, _ = load_already_done_stats(OUTPUT_CSV)
    print("\n" + "=" * 80, flush=True)
    print(f"Pipeline Completed! Total rows in {OUTPUT_CSV}: {len(total_done_final)}", flush=True)
    print(f"Flagged/dropped chunks logged in: {MISMATCH_CSV}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_pipeline()
