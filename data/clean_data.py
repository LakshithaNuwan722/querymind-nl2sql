"""
Day 2 - Step 3: Clean Spider Dataset
Goal: Remove bad samples, fix issues
"""

import os
import re
import json
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

# ─────────────────────────────────────────
# LOAD SCHEMA LOOKUP
# ─────────────────────────────────────────

try:
    with open("data/raw/schema_lookup.json", "r") as f:
        SCHEMA_LOOKUP = json.load(f)
    print(f"   ✅ Schema lookup loaded: {len(SCHEMA_LOOKUP)} databases")
except:
    SCHEMA_LOOKUP = {}
    print("   ⚠️  Schema lookup not found")

load_dotenv()

print("=" * 60)
print("STEP 3: CLEANING SPIDER DATASET")
print("=" * 60)


# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────

print("\n📥 Loading Spider dataset...")
dataset    = load_dataset("spider")
train_data = dataset["train"]
val_data   = dataset["validation"]

print(f"   Train samples before cleaning : {len(train_data)}")
print(f"   Val samples before cleaning   : {len(val_data)}")


# ─────────────────────────────────────────
# 2. DEFINE CLEANING FUNCTIONS
# ─────────────────────────────────────────

def is_valid_sql(sql: str) -> bool:
    """
    Check if SQL query is valid
    Basic checks only
    """
    if not sql or not sql.strip():
        return False
    
    sql_upper = sql.upper().strip()
    
    # Must start with SELECT
    if not sql_upper.startswith("SELECT"):
        return False
    
    # Must have FROM
    if "FROM" not in sql_upper:
        return False
    
    # Must not be too short
    if len(sql.strip()) < 10:
        return False
    
    # Must not be too long (over 500 chars = too complex)
    if len(sql.strip()) > 500:
        return False
    
    return True


def is_valid_question(question: str) -> bool:
    """
    Check if question is valid
    """
    if not question or not question.strip():
        return False
    
    # Must be at least 5 words
    words = question.strip().split()
    if len(words) < 3:
        return False
    
    # Must not be too long
    if len(words) > 60:
        return False
    
    # Must be English (basic check)
    # Check if mostly ASCII characters
    ascii_ratio = sum(1 for c in question if ord(c) < 128) / len(question)
    if ascii_ratio < 0.8:
        return False
    
    return True


def clean_sql(sql: str) -> str:
    """
    Clean and normalize SQL query
    """
    # Remove extra whitespace
    sql = " ".join(sql.split())
    
    # Remove trailing semicolon
    sql = sql.rstrip(";").strip()
    
    # Uppercase SQL keywords
    keywords = [
        "SELECT", "FROM", "WHERE", "AND", "OR",
        "GROUP BY", "ORDER BY", "HAVING", "LIMIT",
        "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN",
        "ON", "AS", "DISTINCT", "COUNT", "SUM",
        "AVG", "MAX", "MIN", "NOT", "IN", "LIKE",
        "BETWEEN", "IS", "NULL", "DESC", "ASC",
        "UNION", "INTERSECT", "EXCEPT"
    ]
    
    for keyword in keywords:
        # Replace keyword with uppercase version
        # Use regex to match whole words only
        pattern = r'\b' + keyword + r'\b'
        sql = re.sub(pattern, keyword, sql, flags=re.IGNORECASE)
    
    return sql.strip()


def clean_question(question: str) -> str:
    """
    Clean question text
    """
    # Remove extra whitespace
    question = " ".join(question.split())
    
    # Capitalize first letter
    question = question.strip()
    if question:
        question = question[0].upper() + question[1:]
    
    # Add question mark if missing
    if question and not question.endswith("?"):
        question = question + "?"
    
    return question


def build_schema_string(sample) -> str:
    """
    Get schema string from lookup
    """
    db_id = sample['db_id']
    
    # Get from lookup
    if db_id in SCHEMA_LOOKUP:
        return SCHEMA_LOOKUP[db_id]
    
    # Fallback
    return f"Database: {db_id}"


# ─────────────────────────────────────────
# 3. CLEAN TRAINING DATA
# ─────────────────────────────────────────

print("\n🧹 Cleaning training data...")

cleaned_train = []
rejected      = {
    "invalid_sql"      : 0,
    "invalid_question" : 0,
    "empty_schema"     : 0,
    "duplicates"       : 0,
}

seen_questions = set()

for i, sample in enumerate(train_data):
    
    question = sample['question']
    sql      = sample['query']
    
    # Check for duplicate questions
    q_lower = question.lower().strip()
    if q_lower in seen_questions:
        rejected["duplicates"] += 1
        continue
    seen_questions.add(q_lower)
    
    # Validate question
    if not is_valid_question(question):
        rejected["invalid_question"] += 1
        continue
    
    # Validate SQL
    if not is_valid_sql(sql):
        rejected["invalid_sql"] += 1
        continue
    
    # Build schema
    schema = build_schema_string(sample)
    if not schema.strip():
        rejected["empty_schema"] += 1
        continue
    
    # Clean question and SQL
    clean_q   = clean_question(question)
    clean_s   = clean_sql(sql)
    
    # Add to cleaned data
    cleaned_train.append({
        "db_id"    : sample['db_id'],
        "question" : clean_q,
        "sql"      : clean_s,
        "schema"   : schema,
    })


# ─────────────────────────────────────────
# 4. CLEAN VALIDATION DATA
# ─────────────────────────────────────────

print("🧹 Cleaning validation data...")

cleaned_val = []

for sample in val_data:
    
    question = sample['question']
    sql      = sample['query']
    
    if not is_valid_question(question):
        continue
    if not is_valid_sql(sql):
        continue
    
    schema = build_schema_string(sample)
    if not schema.strip():
        continue
    
    clean_q = clean_question(question)
    clean_s = clean_sql(sql)
    
    cleaned_val.append({
        "db_id"    : sample['db_id'],
        "question" : clean_q,
        "sql"      : clean_s,
        "schema"   : schema,
    })


# ─────────────────────────────────────────
# 5. PRINT CLEANING RESULTS
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CLEANING RESULTS")
print("=" * 60)

print(f"\n📊 Training Data:")
print(f"   Before cleaning : {len(train_data)}")
print(f"   After cleaning  : {len(cleaned_train)}")
print(f"   Removed         : {len(train_data) - len(cleaned_train)}")

print(f"\n📊 Rejection Reasons:")
for reason, count in rejected.items():
    print(f"   {reason:<20} : {count}")

print(f"\n📊 Validation Data:")
print(f"   Before cleaning : {len(val_data)}")
print(f"   After cleaning  : {len(cleaned_val)}")


# ─────────────────────────────────────────
# 6. SHOW SAMPLE CLEANED DATA
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SAMPLE CLEANED DATA")
print("=" * 60)

for i in range(3):
    item = cleaned_train[i]
    print(f"\n--- Cleaned Sample {i+1} ---")
    print(f"DB       : {item['db_id']}")
    print(f"Question : {item['question']}")
    print(f"SQL      : {item['sql']}")
    print(f"Schema   :\n{item['schema']}")


# ─────────────────────────────────────────
# 7. SAVE CLEANED DATA
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SAVING CLEANED DATA")
print("=" * 60)

# Save as JSON files
with open("data/processed/cleaned_train.json", "w") as f:
    json.dump(cleaned_train, f, indent=2)

with open("data/processed/cleaned_val.json", "w") as f:
    json.dump(cleaned_val, f, indent=2)

print(f"\n✅ Saved cleaned_train.json ({len(cleaned_train)} samples)")
print(f"✅ Saved cleaned_val.json   ({len(cleaned_val)} samples)")


# ─────────────────────────────────────────
# 8. SAVE CLEANING REPORT
# ─────────────────────────────────────────

report = {
    "original_train"  : len(train_data),
    "cleaned_train"   : len(cleaned_train),
    "original_val"    : len(val_data),
    "cleaned_val"     : len(cleaned_val),
    "rejection_reasons": rejected,
}

with open("data/processed/cleaning_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"✅ Saved cleaning_report.json")
print("\n→ Next: Run data/format_data.py")