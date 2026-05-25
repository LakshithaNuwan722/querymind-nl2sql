"""
Day 8 - Load fine-tuned model from HuggingFace
Test it works locally
"""

import os
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

# 🚀 Force Unsloth to use standard, stable HuggingFace download engine (essential for VPN/WARP and Windows proxies)
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"

# 🚀 Force high-speed HuggingFace download tool (hf_transfer)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("=" * 60)
print("LOADING FINE-TUNED MODEL")
print("=" * 60)

# Check Token
token = os.environ.get("HF_TOKEN")
if token:
    print(f"🔑 HF Token detected successfully! (Starts with: {token[:8]}...)")
else:
    print("⚠️  Warning: HF_TOKEN not found in .env file. Download might be throttled.")

# ─────────────────────────────────────────
# 1. CHECK GPU
# ─────────────────────────────────────────

import torch
print("\n🖥️  System Check:")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory
    print(f"   ✅ GPU   : {gpu_name}")
    print(f"   ✅ VRAM  : {gpu_mem/(1024**3):.1f} GB")
    DEVICE = "cuda"
else:
    print(f"   ⚠️  No GPU - using CPU (slower)")
    DEVICE = "cpu"


# ─────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────

print("\n📥 Loading model from HuggingFace...")
print("   First time = downloads (~500MB)")
print("   Next time  = loads from cache")

MODEL_NAME = "lakshitha722/querymind-nl2sql"

try:
    # Try with Unsloth (faster)
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = 1024,
        load_in_4bit   = True,
        dtype          = None,
    )

    FastLanguageModel.for_inference(model)
    print(f"\n   ✅ Model loaded with Unsloth!")
    LOADED_WITH = "unsloth"

except Exception as e:
    print(f"\n   ⚠️  Unsloth failed: {e}")
    print(f"   Trying HuggingFace transformers...")

    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer
    from peft import PeftModel

    # Load base model
    base_model_name = "unsloth/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # On CPU, bitsandbytes 4-bit quantization isn't natively supported.
    # Fallback gracefully to fp32 or fp16 if on CPU.
    if DEVICE == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit    = True,
            device_map      = "auto",
        )
    else:
        print("   ⚠️  Running on CPU: Loading in standard float32 (bitsandbytes/4-bit requires a GPU)")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map      = "cpu",
        )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, MODEL_NAME)
    model.eval()

    print(f"\n   ✅ Model loaded with transformers!")
    LOADED_WITH = "transformers"


# ─────────────────────────────────────────
# 3. INFERENCE FUNCTION
# ─────────────────────────────────────────

INFERENCE_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Convert the following natural language question to a SQL query based on the given database schema. Return ONLY the SQL query, nothing else.

### Schema:
{schema}

### Question:
{question}

### Response:
"""

def generate_sql(
    question    : str,
    schema      : str,
    max_tokens  : int = 150,
) -> dict:
    """
    Generate SQL from natural language question

    Args:
        question  : Natural language question
        schema    : Database schema string
        max_tokens: Max tokens to generate

    Returns:
        dict with sql, tokens, latency
    """
    import time

    prompt = INFERENCE_TEMPLATE.format(
        schema   = schema,
        question = question,
    )

    inputs = tokenizer(
        [prompt],
        return_tensors = "pt",
    ).to(DEVICE)

    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_tokens,
            temperature    = 0.1,
            do_sample      = False,
            pad_token_id   = tokenizer.eos_token_id,
        )

    latency_ms = (time.time() - start) * 1000

    # Decode output
    input_length = inputs['input_ids'].shape[1]
    generated    = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens = True,
    ).strip()

    # Clean output
    # Take only first line (the SQL)
    sql = generated.split('\n')[0].strip()
    sql = sql.rstrip(';').strip()

    return {
        "sql"        : sql,
        "latency_ms" : round(latency_ms, 1),
        "tokens"     : outputs.shape[1] - input_length,
    }


# ─────────────────────────────────────────
# 4. RUN TEST PREDICTIONS
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("TESTING MODEL PREDICTIONS")
print("=" * 60)

test_cases = [
    {
        "question" : "How many employees are there?",
        "schema"   : "Database: company\nTables: employees (id, name, dept, salary)",
        "expected" : "SELECT COUNT(*) FROM employees",
    },
    {
        "question" : "What is the average salary?",
        "schema"   : "Database: hr\nTables: employees (id, name, salary, dept)",
        "expected" : "SELECT AVG(salary) FROM employees",
    },
    {
        "question" : "List all departments ordered by name",
        "schema"   : "Database: company\nTables: departments (id, name, budget)",
        "expected" : "SELECT * FROM departments ORDER BY name",
    },
    {
        "question" : "How many orders were placed by each customer?",
        "schema"   : "Database: sales\nTables: orders (id, customer_id, amount, date)",
        "expected" : "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
    },
    {
        "question" : "Find the maximum salary in each department",
        "schema"   : "Database: hr\nTables: employees (id, name, salary, department)",
        "expected" : "SELECT department, MAX(salary) FROM employees GROUP BY department",
    },
]

print(f"\nRunning {len(test_cases)} test predictions...\n")

results = []
for i, test in enumerate(test_cases, 1):

    result = generate_sql(
        question = test["question"],
        schema   = test["schema"],
    )

    results.append({
        **test,
        **result,
    })

    print(f"Test {i}:")
    print(f"   Question : {test['question']}")
    print(f"   Expected : {test['expected']}")
    print(f"   Got      : {result['sql']}")
    print(f"   Latency  : {result['latency_ms']} ms")
    print()


# ─────────────────────────────────────────
# 5. SUMMARY
# ─────────────────────────────────────────

print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)

avg_latency = sum(r['latency_ms'] for r in results) / len(results)
avg_tokens  = sum(r['tokens'] for r in results) / len(results)

print(f"\n   Model        : {MODEL_NAME}")
print(f"   Loaded with  : {LOADED_WITH}")
print(f"   Device       : {DEVICE}")
print(f"   Avg latency  : {avg_latency:.0f} ms")
print(f"   Avg tokens   : {avg_tokens:.0f}")
print(f"\n   ✅ Model working correctly!")
print(f"\n→ Next: Run evaluation/run_evaluation.py")
