"""  
Day 9 - Full 3-Way Evaluation  
Compare: Base Model vs Fine-tuned vs GPT-4  
"""  

import os  
import json  
import time  
import random  
import torch  
from dotenv import load_dotenv  

# Load environmental variables
load_dotenv()  
random.seed(42)  

# 🚀 Force Unsloth to use standard, stable HuggingFace download engine (essential for VPN/WARP and Windows proxies)
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("=" * 60)  
print("3-WAY MODEL EVALUATION")  
print("=" * 60)  
print("\nModels:")  
print("   1. Base LLaMA 8B     (via Groq API)")  
print("   2. Fine-tuned LLaMA  (local model)")  
print("   3. LLaMA 70B         (via Groq API)")  
  
  
# ─────────────────────────────────────────  
# 1. LOAD TEST DATA  
# ─────────────────────────────────────────  
  
print("\n📥 Loading test data...")  
  
def load_jsonl(filepath):  
    data = []  
    with open(filepath, "r", encoding="utf-8") as f:  
        for line in f:  
            line = line.strip()  
            if line:  
                data.append(json.loads(line))  
    return data  

# Smart filepath detection to avoid FileNotFoundError
filepath = "data/splits/test.jsonl"
if not os.path.exists(filepath):
    # Try checking relative to the directory of this file
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    alt_path = os.path.join(script_dir, "..", "data", "splits", "test.jsonl")
    if os.path.exists(alt_path):
        filepath = alt_path
    else:
        alt_path_2 = os.path.join(script_dir, "data", "splits", "test.jsonl")
        if os.path.exists(alt_path_2):
            filepath = alt_path_2

try:
    test_data = load_jsonl(filepath)  
except FileNotFoundError:
    print(f"   ❌ Error: Could not find 'data/splits/test.jsonl' at {os.path.abspath(filepath)}")
    print("      Please make sure you are running the script from the root directory, or check your path.")
    exit(1)

random.shuffle(test_data)  
  
# Use 100 samples for evaluation  
EVAL_SIZE    = 100  
eval_samples = test_data[:EVAL_SIZE]  
  
print(f"   ✅ Total test: {len(test_data)}")  
print(f"   ✅ Evaluating: {EVAL_SIZE} samples")  
  
  
# ─────────────────────────────────────────  
# 2. LOAD FINE-TUNED MODEL  
# ─────────────────────────────────────────  
  
print("\n📥 Loading fine-tuned model...")  
  
MODEL_NAME = "lakshitha722/querymind-nl2sql"  
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"  
  
try:  
    from unsloth import FastLanguageModel  
  
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(  
        model_name     = MODEL_NAME,  
        max_seq_length = 1024,  
        load_in_4bit   = True,  
        dtype          = None,  
    )  
    FastLanguageModel.for_inference(ft_model)  
    print(f"   ✅ Fine-tuned model loaded successfully on {DEVICE.upper()}!")  
  
except Exception as e:  
    print(f"   ❌ Failed to load model: {e}")  
    exit(1)  
  
  
# ─────────────────────────────────────────  
# 3. SETUP GROQ CLIENT  
# ─────────────────────────────────────────  
  
print("\n🔌 Setting up Groq client...")  

try:
    from groq import Groq  
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))  
    print("   ✅ Groq ready")  
except ImportError:
    print("   ❌ Error: 'groq' package not found. Run 'pip install groq' to install it.")
    exit(1)
except Exception as e:
    print(f"   ❌ Error initializing Groq client: {e}")
    exit(1)
  
  
# ─────────────────────────────────────────  
# 4. PROMPT TEMPLATES  
# ─────────────────────────────────────────  
  
# Template for base models (via Groq)  
GROQ_PROMPT = """Convert this natural language question to a SQL query.  
Return ONLY the SQL query, nothing else.  
  
Schema:  
{schema}  
  
Question: {question}  
  
SQL:"""  
  
# Template for fine-tuned model  
FT_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.  
  
### Instruction:  
Convert the following natural language question to a SQL query based on the given database schema. Return ONLY the SQL query, nothing else.  
  
### Schema:  
{schema}  
  
### Question:  
{question}  
  
### Response:  
"""  
  
  
# ─────────────────────────────────────────  
# 5. EVALUATION FUNCTIONS  
# ─────────────────────────────────────────  
  
def clean_sql(raw: str) -> str:  
    """Clean SQL output from model"""  
    import re  
  
    if not raw:  
        return ""  
  
    # Remove code blocks  
    cleaned = re.sub(r'```sql\s*', '', raw, flags=re.IGNORECASE)  
    cleaned = re.sub(r'```\s*',    '', cleaned)  
  
    # Take first meaningful line  
    lines = cleaned.strip().split('\n')  
    sql_lines = []  
  
    for line in lines:  
        line = line.strip()  
        if not line:  
            continue  
        # Stop at explanation text  
        if sql_lines and re.match(  
            r'^(This|The|Note|Here|Result|Output)',  
            line, re.IGNORECASE  
        ):  
            break  
        sql_lines.append(line)  
  
    result = ' '.join(sql_lines).strip()  
    result = result.rstrip(';').strip()  
    result = result.strip('"').strip("'")  
  
    return result  
  
  
def is_valid_sql(sql: str) -> bool:  
    """Check basic SQL validity"""  
    if not sql or len(sql.strip()) < 5:  
        return False  
    sql_upper = sql.upper()  
    return "SELECT" in sql_upper and "FROM" in sql_upper  
  
  
def calc_similarity(gen: str, ref: str) -> float:  
    """Calculate SQL similarity score"""  
    import re  
  
    if not gen or not ref:  
        return 0.0  
  
    gen_upper = gen.upper()  
    ref_upper = ref.upper()  
  
    keywords = [  
        "SELECT", "FROM", "WHERE", "GROUP BY",  
        "ORDER BY", "HAVING", "JOIN", "COUNT",  
        "SUM", "AVG", "MAX", "MIN", "DISTINCT",  
    ]  
  
    gen_kw = set(kw for kw in keywords if kw in gen_upper)  
    ref_kw = set(kw for kw in keywords if kw in ref_upper)  
  
    if not ref_kw:  
        return 0.0  
  
    intersection = gen_kw & ref_kw  
    union        = gen_kw | ref_kw  
    kw_score     = len(intersection) / len(union) if union else 0  
  
    # Table name match  
    def get_tables(sql):  
        tables = set()  
        tables.update(re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE))  
        tables.update(re.findall(r'JOIN\s+(\w+)',  sql, re.IGNORECASE))  
        return {t.lower() for t in tables}  
  
    gen_tables = get_tables(gen)  
    ref_tables = get_tables(ref)  
  
    if ref_tables:  
        table_score = len(gen_tables & ref_tables) / len(ref_tables)  
    else:  
        table_score = 1.0  
  
    return round((kw_score * 0.6) + (table_score * 0.4), 3)  
  
  
def generate_groq(  
    question   : str,  
    schema     : str,  
    model_name : str,  
) -> tuple:  
    """Generate SQL using Groq API"""  
    prompt = GROQ_PROMPT.format(  
        schema   = schema,  
        question = question,  
    )  
  
    start = time.time()  
    try:  
        response = groq_client.chat.completions.create(  
            model       = model_name,  
            messages    = [{"role": "user", "content": prompt}],  
            max_tokens  = 200,  
            temperature = 0,  
        )  
        latency   = (time.time() - start) * 1000  
        generated = response.choices[0].message.content  
        tokens    = response.usage.total_tokens  
        return generated, round(latency, 1), tokens  
  
    except Exception as e:  
        latency = (time.time() - start) * 1000  
        return "", round(latency, 1), 0  
  
  
def generate_finetuned(  
    question : str,  
    schema   : str,  
) -> tuple:  
    """Generate SQL using fine-tuned model"""  
    prompt = FT_PROMPT.format(  
        schema   = schema,  
        question = question,  
    )  
  
    inputs = ft_tokenizer(  
        [prompt],  
        return_tensors = "pt",  
    ).to(DEVICE)  
  
    start = time.time()  
  
    with torch.no_grad():  
        outputs = ft_model.generate(  
            **inputs,  
            max_new_tokens = 150,  
            temperature    = 0.1,  
            do_sample      = False,  
            pad_token_id   = ft_tokenizer.eos_token_id,  
        )  
  
    latency      = (time.time() - start) * 1000  
    input_length = inputs['input_ids'].shape[1]  
    generated    = ft_tokenizer.decode(  
        outputs[0][input_length:],  
        skip_special_tokens = True,  
    ).strip()  
  
    tokens = outputs.shape[1] - input_length  
  
    return generated, round(latency, 1), tokens  
  
  
# ─────────────────────────────────────────  
# 6. RUN EVALUATION FOR ALL MODELS  
# ─────────────────────────────────────────  
  
MODELS = {  
    "base_8b"    : {  
        "name"     : "llama-3.1-8b-instant",  
        "type"     : "groq",  
        "label"    : "Base LLaMA 8B",  
    },  
    "finetuned"  : {  
        "name"     : MODEL_NAME,  
        "type"     : "local",  
        "label"    : "Fine-tuned LLaMA 3B",  
    },  
    "base_70b"   : {  
        "name"     : "llama-3.3-70b-versatile",  
        "type"     : "groq",  
        "label"    : "Base LLaMA 70B",  
    },  
}  
  
all_results = {}  
  
for model_key, model_info in MODELS.items():  
  
    print(f"\n{'='*60}")  
    print(f"EVALUATING: {model_info['label']}")  
    print(f"{'='*60}")  
  
    predictions = []  
    latencies   = []  
    total_tok   = 0  
    errors      = 0  
  
    for i, sample in enumerate(eval_samples):  
  
        question  = sample.get("question", "")  
        schema    = sample.get("schema", "")  
        reference = sample.get("sql", "")  
  
        # Progress  
        if (i + 1) % 20 == 0:  
            valid_now = sum(  
                1 for p in predictions if p['valid']  
            )  
            print(  
                f"   [{i+1:>3}/{EVAL_SIZE}] "  
                f"Valid: {valid_now}/{i+1} "  
                f"({valid_now/(i+1)*100:.0f}%)"  
            )  
  
        # Generate  
        if model_info["type"] == "groq":  
            raw, latency, tokens = generate_groq(  
                question   = question,  
                schema     = schema,  
                model_name = model_info["name"],  
            )  
            time.sleep(0.3)  # Rate limit  
  
        else:  
            raw, latency, tokens = generate_finetuned(  
                question = question,  
                schema   = schema,  
            )  
  
        latencies.append(latency)  
        total_tok += tokens  
  
        if not raw:  
            errors += 1  
  
        # Clean and evaluate  
        sql        = clean_sql(raw)  
        valid      = is_valid_sql(sql)  
        similarity = calc_similarity(sql, reference)  
        exact      = sql.upper().strip() == reference.upper().strip()  
  
        predictions.append({  
            "question"   : question,  
            "reference"  : reference,  
            "generated"  : sql,  
            "valid"      : valid,  
            "exact"      : exact,  
            "similarity" : similarity,  
            "latency_ms" : latency,  
        })  
  
    # Calculate metrics  
    total      = len(predictions)  
    valid_pct  = sum(1 for p in predictions if p['valid']) / total * 100  
    exact_pct  = sum(1 for p in predictions if p['exact']) / total * 100  
    avg_sim    = sum(p['similarity'] for p in predictions) / total  
    avg_lat    = sum(latencies) / len(latencies)  
    high_sim   = sum(1 for p in predictions if p['similarity'] >= 0.7)  
    high_sim_pct = (high_sim / total) * 100
  
    summary = {  
        "label"          : model_info['label'],  
        "total"          : total,  
        "valid_sql_pct"  : round(valid_pct, 1),  
        "exact_match_pct": round(exact_pct, 1),  
        "avg_similarity" : round(avg_sim, 3),  
        "high_sim_count" : high_sim,  
        "high_sim_pct"   : round(high_sim_pct, 1),  
        "avg_latency_ms" : round(avg_lat, 1),  
        "total_tokens"   : total_tok,  
        "avg_tokens"     : round(total_tok / total, 1),  
        "errors"         : errors,  
    }  
  
    all_results[model_key] = {  
        "summary"     : summary,  
        "predictions" : predictions,  
    }  
  
    # Save intermediate  
    os.makedirs("evaluation/results", exist_ok=True)  
    with open(f"evaluation/results/eval_{model_key}.json", "w") as f:  
        json.dump(all_results[model_key], f, indent=2)  
  
    print(f"\n   📊 Results:")  
    print(f"   Valid SQL      : {valid_pct:.1f}%")  
    print(f"   Exact Match    : {exact_pct:.1f}%")  
    print(f"   Avg Similarity : {avg_sim:.3f}")  
    print(f"   High Sim (>=0.7): {high_sim_pct:.1f}%")  
    print(f"   Avg Latency    : {avg_lat:.0f} ms")  
  
  
# ─────────────────────────────────────────  
# 7. COMPARISON TABLE  
# ─────────────────────────────────────────  
  
print(f"\n{'='*60}")  
print(f"FINAL COMPARISON TABLE")  
print(f"{'='*60}\n")  
  
metrics = [  
    ("Valid SQL %",      "valid_sql_pct"),  
    ("Exact Match %",    "exact_match_pct"),  
    ("Avg Similarity",   "avg_similarity"),  
    ("High Sim % (>=0.7)","high_sim_pct"),  
    ("Avg Latency (ms)", "avg_latency_ms"),  
    ("Avg Tokens",       "avg_tokens"),  
]  
  
# Header  
print(f"{'Metric':<22}", end="")  
for key, info in MODELS.items():  
    print(f"{info['label']:<22}", end="")  
print()  
print("─" * (22 + 22 * len(MODELS)))  
  
# Rows  
for metric_name, metric_key in metrics:  
    print(f"{metric_name:<22}", end="")  
    for key in MODELS:  
        val = all_results[key]["summary"].get(metric_key, "N/A")  
        print(f"{str(val):<22}", end="")  
    print()  
  
# Cost row  
print(f"{'Cost/1k queries':<22}", end="")  
costs = {"base_8b": "$0.00", "finetuned": "$0.00", "base_70b": "$0.00"}  
for key in MODELS:  
    print(f"{costs[key]:<22}", end="")  
print()  
  
  
# ─────────────────────────────────────────  
# 8. SAVE COMBINED RESULTS  
# ─────────────────────────────────────────  
  
combined = {  
    "eval_samples"  : EVAL_SIZE,  
    "models"        : MODELS,  
    "results"       : {  
        k: v["summary"]  
        for k, v in all_results.items()  
    },  
}  
  
with open("evaluation/results/eval_combined.json", "w") as f:  
    json.dump(combined, f, indent=2)  
  
print(f"\n✅ Results saved to evaluation/results/")  
print(f"\n→ Next: Run evaluation/create_charts.py")
