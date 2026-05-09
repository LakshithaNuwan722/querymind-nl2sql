"""
Baseline Testing Configuration
"""

# ─────────────────────────────────────────
# MODELS TO TEST
# ─────────────────────────────────────────

MODELS = {
    "llama_8b" : {
        "name"        : "llama-3.1-8b-instant",
        "provider"    : "groq",
        "description" : "Llama 3.1 8B - Small fast model",
        "max_tokens"  : 200,
        "temperature" : 0,
    },
    "llama_70b" : {
        "name"        : "llama-3.3-70b-versatile",
        "provider"    : "groq",
        "description" : "Llama 3.3 70B - Large powerful model",
        "max_tokens"  : 200,
        "temperature" : 0,
    },
}

# ─────────────────────────────────────────
# TEST SETTINGS
# ─────────────────────────────────────────

# Number of test samples to use
# Using 100 from test.jsonl
TEST_SAMPLE_SIZE = 100

# Random seed for reproducibility
RANDOM_SEED = 42

# Paths
TEST_DATA_PATH    = "data/splits/test.jsonl"
RESULTS_SAVE_PATH = "evaluation/baseline/"

# ─────────────────────────────────────────
# PROMPT TEMPLATE FOR BASELINE
# ─────────────────────────────────────────

# Same template used in training data
BASELINE_PROMPT = """### Instruction:
Convert the following natural language question to a SQL query based on the given database schema. Return ONLY the SQL query, nothing else.

### Schema:
{schema}

### Question:
{question}

### SQL:
"""