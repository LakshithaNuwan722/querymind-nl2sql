"""
Day 1 - Step 7: Spider Dataset Exploration
Goal: Understand the dataset before cleaning
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("QUERYMIND - Dataset Exploration")
print("=" * 60)


# ─────────────────────────────────────────
# 1. LOAD SPIDER DATASET
# ─────────────────────────────────────────

print("\n📥 Loading Spider dataset from Hugging Face...")

dataset = load_dataset("spider")

print("\n✅ Dataset loaded successfully!")
print(f"   Training samples   : {len(dataset['train'])}")
print(f"   Validation samples : {len(dataset['validation'])}")


# ─────────────────────────────────────────
# 2. LOOK AT DATASET STRUCTURE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)

# See what columns/features exist
print("\n📋 Available columns:")
for col in dataset['train'].column_names:
    print(f"   → {col}")


# ─────────────────────────────────────────
# 3. LOOK AT FIRST 5 SAMPLES IN DETAIL
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("FIRST 5 SAMPLES - DETAILED VIEW")
print("=" * 60)

for i in range(5):
    sample = dataset['train'][i]
    print(f"\n{'─'*50}")
    print(f"Sample #{i+1}")
    print(f"{'─'*50}")
    
    print(f"\n❓ QUESTION:")
    print(f"   {sample['question']}")
    
    print(f"\n🗄️  DATABASE NAME:")
    print(f"   {sample['db_id']}")
    
    print(f"\n📊 SQL QUERY:")
    print(f"   {sample['query']}")
    
    # Schema information
    print(f"\n🏗️  SCHEMA INFO:")
    tables = sample['db_table_names'] if 'db_table_names' in sample else []
    if tables:
        for table in tables:
            print(f"   Table: {table}")


# ─────────────────────────────────────────
# 4. ANALYZE SQL COMPLEXITY
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SQL COMPLEXITY ANALYSIS")
print("=" * 60)

train_data = dataset['train']

# Count SQL keywords to measure complexity
complexity_counts = {
    'SELECT': 0,
    'WHERE': 0,
    'GROUP BY': 0,
    'ORDER BY': 0,
    'JOIN': 0,
    'HAVING': 0,
    'UNION': 0,
    'INTERSECT': 0,
    'EXCEPT': 0,
    'LIMIT': 0,
    'DISTINCT': 0,
    'COUNT': 0,
    'SUM': 0,
    'AVG': 0,
    'MAX': 0,
    'MIN': 0,
}

for sample in train_data:
    sql_upper = sample['query'].upper()
    for keyword in complexity_counts:
        if keyword in sql_upper:
            complexity_counts[keyword] += 1

total = len(train_data)
print(f"\n📊 SQL Pattern Distribution (out of {total} samples):")
print(f"\n{'Keyword':<15} {'Count':<10} {'Percentage':<10}")
print("-" * 35)
for keyword, count in complexity_counts.items():
    percentage = (count / total) * 100
    bar = "█" * int(percentage / 5)
    print(f"{keyword:<15} {count:<10} {percentage:.1f}%  {bar}")


# ─────────────────────────────────────────
# 5. ANALYZE QUESTION LENGTHS
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("QUESTION + SQL LENGTH ANALYSIS")
print("=" * 60)

question_lengths = [len(s['question'].split()) for s in train_data]
sql_lengths = [len(s['query'].split()) for s in train_data]

print(f"\n📝 Question Length (words):")
print(f"   Minimum  : {min(question_lengths)}")
print(f"   Maximum  : {max(question_lengths)}")
print(f"   Average  : {sum(question_lengths)/len(question_lengths):.1f}")

print(f"\n💾 SQL Length (words):")
print(f"   Minimum  : {min(sql_lengths)}")
print(f"   Maximum  : {max(sql_lengths)}")
print(f"   Average  : {sum(sql_lengths)/len(sql_lengths):.1f}")


# ─────────────────────────────────────────
# 6. ANALYZE DATABASE DISTRIBUTION
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("DATABASE DISTRIBUTION")
print("=" * 60)

db_counts = {}
for sample in train_data:
    db = sample['db_id']
    db_counts[db] = db_counts.get(db, 0) + 1

# Sort by count
sorted_dbs = sorted(db_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\n🗄️  Top 15 Most Common Databases:")
print(f"\n{'Database':<30} {'Samples':<10}")
print("-" * 40)
for db, count in sorted_dbs[:15]:
    bar = "█" * (count // 5)
    print(f"{db:<30} {count:<10} {bar}")

print(f"\n   Total unique databases: {len(db_counts)}")


# ─────────────────────────────────────────
# 7. LOOK AT SCHEMA STRUCTURE DEEPLY
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("DEEP SCHEMA ANALYSIS - Sample #1")
print("=" * 60)

sample = dataset['train'][0]
print(f"\nDatabase: {sample['db_id']}")
print(f"\nAll available fields in this sample:")
for key, value in sample.items():
    print(f"\n  Key: '{key}'")
    print(f"  Type: {type(value).__name__}")
    print(f"  Value: {str(value)[:200]}")


# ─────────────────────────────────────────
# 8. IDENTIFY POTENTIAL ISSUES
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("POTENTIAL DATA ISSUES")
print("=" * 60)

issues = {
    'empty_questions': 0,
    'empty_sql': 0,
    'very_long_sql': 0,
    'very_long_question': 0,
    'multiple_statements': 0,
}

for sample in train_data:
    if not sample['question'].strip():
        issues['empty_questions'] += 1
    if not sample['query'].strip():
        issues['empty_sql'] += 1
    if len(sample['query'].split()) > 100:
        issues['very_long_sql'] += 1
    if len(sample['question'].split()) > 50:
        issues['very_long_question'] += 1
    if ';' in sample['query'][:-1]:  # semicolon not at end
        issues['multiple_statements'] += 1

print(f"\n⚠️  Issues Found:")
for issue, count in issues.items():
    status = "✅ None" if count == 0 else f"⚠️  {count} found"
    print(f"   {issue:<25} : {status}")


# ─────────────────────────────────────────
# 9. SAVE EXPLORATION SUMMARY
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SAVING EXPLORATION SUMMARY")
print("=" * 60)

summary = {
    "dataset": "Spider",
    "total_train": len(dataset['train']),
    "total_validation": len(dataset['validation']),
    "sql_patterns": {k: v for k, v in complexity_counts.items()},
    "question_length": {
        "min": min(question_lengths),
        "max": max(question_lengths),
        "avg": round(sum(question_lengths)/len(question_lengths), 1)
    },
    "sql_length": {
        "min": min(sql_lengths),
        "max": max(sql_lengths),
        "avg": round(sum(sql_lengths)/len(sql_lengths), 1)
    },
    "unique_databases": len(db_counts),
    "issues_found": issues
}

with open("data/raw/exploration_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Summary saved to: data/raw/exploration_summary.json")
print("\n" + "=" * 60)
print("EXPLORATION COMPLETE!")
print("=" * 60)
print("\nNext step: Run data preparation (Day 2)")