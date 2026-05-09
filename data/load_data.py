"""
Day 2 - Step 2: Load and Understand Spider Dataset
Goal: Load dataset and print detailed structure
"""

import os
import json
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("STEP 2: LOADING SPIDER DATASET")
print("=" * 60)


# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────

print("\n📥 Loading Spider dataset...")
dataset = load_dataset("spider")

train_data = dataset["train"]
val_data   = dataset["validation"]

print(f"\n✅ Loaded successfully!")
print(f"   Train samples      : {len(train_data)}")
print(f"   Validation samples : {len(val_data)}")


# ─────────────────────────────────────────
# 2. PRINT EXACT STRUCTURE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("EXACT DATASET STRUCTURE")
print("=" * 60)

sample = train_data[0]

print(f"\n📌 All Available Keys:")
for key in sample.keys():
    value = sample[key]
    print(f"   → {key} : {type(value).__name__} = {str(value)[:80]}")


# ─────────────────────────────────────────
# 3. SHOW 5 SAMPLES - ONLY USE REAL FIELDS
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("5 SAMPLE QUESTION-SQL PAIRS")
print("=" * 60)

for i in range(5):
    sample = train_data[i]
    print(f"\n--- Sample {i+1} ---")
    print(f"DB ID    : {sample['db_id']}")
    print(f"Question : {sample['question']}")
    print(f"SQL      : {sample['query']}")


# ─────────────────────────────────────────
# 4. LOAD SPIDER TABLES (Schema Info)
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("LOADING SCHEMA INFORMATION")
print("=" * 60)

# Spider schema is in tables.json
# Download it from HuggingFace datasets cache
# OR load from the dataset itself

try:
    # Try to get schema from dataset features
    print("\n📌 Dataset Features:")
    print(dataset['train'].features)
    
except Exception as e:
    print(f"Features error: {e}")


# ─────────────────────────────────────────
# 5. CHECK IF SCHEMA DATA EXISTS
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECKING SCHEMA AVAILABILITY")
print("=" * 60)

# Check first sample for any nested data
sample = train_data[0]
print(f"\nFull first sample:")
for key, value in sample.items():
    print(f"\n  Key   : {key}")
    print(f"  Type  : {type(value).__name__}")
    print(f"  Value : {str(value)[:200]}")


# ─────────────────────────────────────────
# 6. ANALYZE WHAT WE HAVE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("DATA ANALYSIS WITH AVAILABLE FIELDS")
print("=" * 60)

# SQL complexity analysis using only query field
complexity = {
    "SELECT"   : 0,
    "WHERE"    : 0,
    "GROUP BY" : 0,
    "ORDER BY" : 0,
    "JOIN"     : 0,
    "HAVING"   : 0,
    "COUNT"    : 0,
    "SUM"      : 0,
    "AVG"      : 0,
    "MAX"      : 0,
    "MIN"      : 0,
}

for sample in train_data:
    sql_upper = sample['query'].upper()
    for keyword in complexity:
        if keyword in sql_upper:
            complexity[keyword] += 1

total = len(train_data)
print(f"\n📊 SQL Pattern Distribution ({total} samples):")
print(f"\n{'Keyword':<15} {'Count':<8} {'Pct':<8} {'Bar'}")
print("-" * 50)
for keyword, count in complexity.items():
    pct = (count/total)*100
    bar = "█" * int(pct/5)
    print(f"{keyword:<15} {count:<8} {pct:.1f}%   {bar}")

# Unique databases
db_ids = [s['db_id'] for s in train_data]
unique_dbs = set(db_ids)
print(f"\n📊 Unique Databases: {len(unique_dbs)}")

# Question length
q_lengths = [len(s['question'].split()) for s in train_data]
print(f"\n📊 Question Length (words):")
print(f"   Min : {min(q_lengths)}")
print(f"   Max : {max(q_lengths)}")
print(f"   Avg : {sum(q_lengths)/len(q_lengths):.1f}")

# SQL length
sql_lengths = [len(s['query'].split()) for s in train_data]
print(f"\n📊 SQL Length (words):")
print(f"   Min : {min(sql_lengths)}")
print(f"   Max : {max(sql_lengths)}")
print(f"   Avg : {sum(sql_lengths)/len(sql_lengths):.1f}")


# ─────────────────────────────────────────
# 7. SAVE SUMMARY
# ─────────────────────────────────────────

summary = {
    "train_size"       : len(train_data),
    "val_size"         : len(val_data),
    "unique_databases" : len(unique_dbs),
    "available_fields" : list(train_data[0].keys()),
    "sql_patterns"     : complexity,
    "question_length"  : {
        "min" : min(q_lengths),
        "max" : max(q_lengths),
        "avg" : round(sum(q_lengths)/len(q_lengths), 1),
    },
    "sql_length"       : {
        "min" : min(sql_lengths),
        "max" : max(sql_lengths),
        "avg" : round(sum(sql_lengths)/len(sql_lengths), 1),
    },
}

with open("data/raw/exploration_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Summary saved to data/raw/exploration_summary.json")
print(f"\n→ Next: Run data/clean_data.py")