"""
Day 3 - Run Baseline Testing
Goal: Test base models BEFORE fine-tuning
"""

import os
import sys
import json
import time
import random

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)

from groq import Groq
from dotenv import load_dotenv
from evaluation.baseline.config import (
    MODELS,
    TEST_SAMPLE_SIZE,
    RANDOM_SEED,
    TEST_DATA_PATH,
    RESULTS_SAVE_PATH,
    BASELINE_PROMPT,
)
from evaluation.baseline.sql_validator import (
    evaluate_single_prediction,
    summarize_results,
    clean_sql_output,
)

load_dotenv()
random.seed(RANDOM_SEED)

print("=" * 60)
print("DAY 3: BASELINE TESTING")
print("=" * 60)


# ─────────────────────────────────────────
# 1. SETUP GROQ CLIENT
# ─────────────────────────────────────────

print("\n🔌 Setting up Groq client...")

try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("   ✅ Groq client ready")
except Exception as e:
    print(f"   ❌ Groq setup failed: {e}")
    exit(1)


# ─────────────────────────────────────────
# 2. LOAD TEST DATA
# ─────────────────────────────────────────

print("\n📥 Loading test data...")

def load_jsonl(filepath: str) -> list:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

try:
    all_test_data = load_jsonl(TEST_DATA_PATH)
    print(f"   ✅ Total test samples: {len(all_test_data)}")
except FileNotFoundError:
    print(f"   ❌ Test file not found: {TEST_DATA_PATH}")
    exit(1)

# Sample 100 from test set
random.shuffle(all_test_data)
test_samples = all_test_data[:TEST_SAMPLE_SIZE]

print(f"   ✅ Using {len(test_samples)} samples for baseline")


# ─────────────────────────────────────────
# 3. DEFINE GENERATION FUNCTION
# ─────────────────────────────────────────

def generate_sql(
    question    : str,
    schema      : str,
    model_name  : str,
    max_tokens  : int = 200,
) -> tuple:
    """
    Generate SQL using Groq API

    Returns:
        (generated_sql, latency_ms, tokens_used)
    """
    prompt = BASELINE_PROMPT.format(
        schema   = schema,
        question = question,
    )

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model    = model_name,
            messages = [
                {
                    "role"    : "user",
                    "content" : prompt,
                }
            ],
            max_tokens  = max_tokens,
            temperature = 0,
        )

        latency_ms   = (time.time() - start_time) * 1000
        generated    = response.choices[0].message.content
        tokens_used  = response.usage.total_tokens

        return generated, round(latency_ms, 1), tokens_used

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        print(f"\n   ⚠️  API error: {e}")
        return "", round(latency_ms, 1), 0


# ─────────────────────────────────────────
# 4. RUN BASELINE FOR ONE MODEL
# ─────────────────────────────────────────

def run_model_baseline(
    model_key   : str,
    model_config: dict,
    test_data   : list,
) -> dict:
    """
    Run baseline evaluation for one model

    Returns:
        dict with all results and metrics
    """
    model_name  = model_config["name"]
    description = model_config["description"]
    max_tokens  = model_config["max_tokens"]

    print(f"\n{'='*60}")
    print(f"TESTING: {description}")
    print(f"Model  : {model_name}")
    print(f"{'='*60}")

    predictions  = []
    latencies    = []
    total_tokens = 0
    errors       = 0

    for i, sample in enumerate(test_data):

        question  = sample.get("question", "")
        schema    = sample.get("schema", "")
        reference = sample.get("sql", "")
        db_id     = sample.get("db_id", "")

        # Progress update
        if (i + 1) % 10 == 0:
            valid_so_far = sum(
                1 for p in predictions if p['is_valid']
            )
            print(
                f"   [{i+1:>3}/{len(test_data)}] "
                f"Valid so far: {valid_so_far}/{i+1}"
            )

        # Generate SQL
        generated, latency_ms, tokens = generate_sql(
            question   = question,
            schema     = schema,
            model_name = model_name,
            max_tokens = max_tokens,
        )

        latencies.append(latency_ms)
        total_tokens += tokens

        if not generated:
            errors += 1

        # Evaluate prediction
        result = evaluate_single_prediction(
            generated_sql = generated,
            reference_sql = reference,
            question      = question,
        )

        result["db_id"]      = db_id
        result["latency_ms"] = latency_ms
        result["tokens"]     = tokens

        predictions.append(result)

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    # Calculate summary metrics
    summary = summarize_results(predictions)

    # Add latency and cost metrics
    summary["avg_latency_ms"]  = round(
        sum(latencies) / len(latencies), 1
    )
    summary["min_latency_ms"]  = round(min(latencies), 1)
    summary["max_latency_ms"]  = round(max(latencies), 1)
    summary["total_tokens"]    = total_tokens
    summary["avg_tokens"]      = round(
        total_tokens / len(test_data), 1
    )
    summary["api_errors"]      = errors
    summary["model_name"]      = model_name
    summary["model_key"]       = model_key
    summary["description"]     = description

    # Print results
    print(f"\n   📊 Results for {description}:")
    print(f"   {'─'*40}")
    print(f"   Valid SQL        : {summary['valid_sql_pct']}%")
    print(f"   Exact Match      : {summary['exact_match_pct']}%")
    print(f"   Avg Similarity   : {summary['avg_similarity']}")
    print(f"   Avg Latency      : {summary['avg_latency_ms']} ms")
    print(f"   Total Tokens     : {summary['total_tokens']}")
    print(f"   API Errors       : {summary['api_errors']}")

    return {
        "summary"     : summary,
        "predictions" : predictions,
    }


# ─────────────────────────────────────────
# 5. RUN ALL MODELS
# ─────────────────────────────────────────

print(f"\n🚀 Starting baseline tests...")
print(f"   Models to test : {len(MODELS)}")
print(f"   Samples each   : {TEST_SAMPLE_SIZE}")
print(f"   Total API calls: {len(MODELS) * TEST_SAMPLE_SIZE}")
print(f"\n   ⏱️  Estimated time: 5-10 minutes")

all_results = {}

for model_key, model_config in MODELS.items():
    results = run_model_baseline(
        model_key    = model_key,
        model_config = model_config,
        test_data    = test_samples,
    )
    all_results[model_key] = results

    # Save intermediate results
    save_path = os.path.join(
        RESULTS_SAVE_PATH,
        f"baseline_{model_key}.json"
    )
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n   💾 Saved: {save_path}")


# ─────────────────────────────────────────
# 6. COMPARE ALL MODELS
# ─────────────────────────────────────────

print(f"\n{'='*60}")
print(f"BASELINE COMPARISON RESULTS")
print(f"{'='*60}")

print(f"\n{'Metric':<25}", end="")
for model_key in all_results:
    desc = MODELS[model_key]["description"][:15]
    print(f"{desc:<20}", end="")
print()
print("─" * (25 + 20 * len(all_results)))

metrics_to_show = [
    ("Valid SQL %",      "valid_sql_pct"),
    ("Exact Match %",    "exact_match_pct"),
    ("Avg Similarity",   "avg_similarity"),
    ("Avg Latency (ms)", "avg_latency_ms"),
    ("Avg Tokens",       "avg_tokens"),
]

for metric_name, metric_key in metrics_to_show:
    print(f"{metric_name:<25}", end="")
    for model_key in all_results:
        value = all_results[model_key]["summary"].get(
            metric_key, "N/A"
        )
        print(f"{str(value):<20}", end="")
    print()


# ─────────────────────────────────────────
# 7. SHOW PREDICTION EXAMPLES
# ─────────────────────────────────────────

print(f"\n{'='*60}")
print(f"PREDICTION EXAMPLES")
print(f"{'='*60}")

# Show 5 examples from first model
first_model    = list(all_results.keys())[0]
sample_preds   = all_results[first_model]["predictions"][:5]

for i, pred in enumerate(sample_preds, 1):
    print(f"\n--- Example {i} ---")
    print(f"Question  : {pred['question']}")
    print(f"Reference : {pred['reference_sql']}")
    print(f"Generated : {pred['generated_sql']}")
    print(f"Valid     : {pred['is_valid']}")
    print(f"Similarity: {pred['similarity']}")
    print(f"Result    : {pred['result']}")


# ─────────────────────────────────────────
# 8. SAVE COMBINED RESULTS
# ─────────────────────────────────────────

print(f"\n{'='*60}")
print(f"SAVING RESULTS")
print(f"{'='*60}")

# Combined summary for easy comparison
combined_summary = {
    "test_samples"   : TEST_SAMPLE_SIZE,
    "random_seed"    : RANDOM_SEED,
    "models_tested"  : list(all_results.keys()),
    "results"        : {
        model_key: results["summary"]
        for model_key, results in all_results.items()
    },
    "note" : (
        "These are BASELINE results BEFORE fine-tuning. "
        "Fine-tuned model results will be added later."
    )
}

combined_path = os.path.join(
    RESULTS_SAVE_PATH,
    "baseline_combined.json"
)

with open(combined_path, "w") as f:
    json.dump(combined_summary, f, indent=2)

print(f"\n   ✅ Combined results: {combined_path}")


# ─────────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────────

print(f"\n{'='*60}")
print(f"DAY 3 COMPLETE!")
print(f"{'='*60}")

print(f"\n📊 Baseline Numbers (Save These!):")
print(f"{'─'*50}")

for model_key, results in all_results.items():
    s = results["summary"]
    print(f"\n   {s['description']}:")
    print(f"   Valid SQL      : {s['valid_sql_pct']}%")
    print(f"   Exact Match    : {s['exact_match_pct']}%")
    print(f"   Avg Similarity : {s['avg_similarity']}")
    print(f"   Avg Latency    : {s['avg_latency_ms']} ms")

print(f"\n{'─'*50}")
print(f"\n   These numbers = BEFORE fine-tuning")
print(f"   After fine-tuning these should improve!")
print(f"\n   Results saved to: evaluation/baseline/")
print(f"\n   → Next: Day 4 - Fine-tuning on Google Colab")