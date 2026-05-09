"""
Day 2 - Step 4: Format Data for Training
Goal: Convert cleaned data into training template format
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("STEP 4: FORMATTING DATA FOR TRAINING")
print("=" * 60)


# ─────────────────────────────────────────
# 1. LOAD CLEANED DATA
# ─────────────────────────────────────────

print("\n📥 Loading cleaned data...")

try:
    with open("data/processed/cleaned_train.json", "r") as f:
        cleaned_train = json.load(f)

    with open("data/processed/cleaned_val.json", "r") as f:
        cleaned_val = json.load(f)

    print(f"   ✅ Train samples : {len(cleaned_train)}")
    print(f"   ✅ Val samples   : {len(cleaned_val)}")

except FileNotFoundError as e:
    print(f"   ❌ File not found: {e}")
    print(f"   → Run data/clean_data.py first")
    exit(1)


# ─────────────────────────────────────────
# 2. SHOW SAMPLE CLEANED DATA STRUCTURE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CLEANED DATA STRUCTURE")
print("=" * 60)

sample = cleaned_train[0]
print(f"\n📌 Keys in cleaned data:")
for key in sample.keys():
    print(f"   → {key} : {str(sample[key])[:80]}")


# ─────────────────────────────────────────
# 3. DEFINE PROMPT TEMPLATES
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("DEFINING PROMPT TEMPLATES")
print("=" * 60)

# Training template - includes SQL answer
PROMPT_TEMPLATE_TRAIN = """### Instruction:
Convert the following natural language question to a SQL query based on the given database schema. Return ONLY the SQL query, nothing else.

### Schema:
{schema}

### Question:
{question}

### SQL:
{sql}"""


# Inference template - no SQL answer
# Used when generating predictions
PROMPT_TEMPLATE_INFERENCE = """### Instruction:
Convert the following natural language question to a SQL query based on the given database schema. Return ONLY the SQL query, nothing else.

### Schema:
{schema}

### Question:
{question}

### SQL:
"""

print("\n✅ Training template:")
print("─" * 40)
print(PROMPT_TEMPLATE_TRAIN[:200] + "...")

print("\n✅ Inference template:")
print("─" * 40)
print(PROMPT_TEMPLATE_INFERENCE[:200] + "...")

# Save templates
templates = {
    "training"  : PROMPT_TEMPLATE_TRAIN,
    "inference" : PROMPT_TEMPLATE_INFERENCE,
    "version"   : "v1",
    "created"   : "Day 2",
}

os.makedirs("data/processed", exist_ok=True)

with open("data/processed/prompt_templates.json", "w") as f:
    json.dump(templates, f, indent=2)

print("\n✅ Templates saved to data/processed/prompt_templates.json")


# ─────────────────────────────────────────
# 4. FORMAT FUNCTIONS
# ─────────────────────────────────────────

def format_sample_train(item: dict) -> dict:
    """
    Format single sample for training
    Includes the SQL answer

    Args:
        item: dict with question, sql, schema, db_id

    Returns:
        dict with formatted text + metadata
    """
    # Build formatted text
    formatted_text = PROMPT_TEMPLATE_TRAIN.format(
        schema   = item.get("schema", f"Database: {item.get('db_id', 'unknown')}"),
        question = item.get("question", ""),
        sql      = item.get("sql", ""),
    )

    return {
        "text"     : formatted_text,
        "db_id"    : item.get("db_id", ""),
        "question" : item.get("question", ""),
        "sql"      : item.get("sql", ""),
        "schema"   : item.get("schema", ""),
    }


def format_sample_inference(item: dict) -> dict:
    """
    Format single sample for inference
    Does NOT include SQL answer

    Args:
        item: dict with question, schema, db_id

    Returns:
        dict with formatted prompt + ground truth sql
    """
    formatted_prompt = PROMPT_TEMPLATE_INFERENCE.format(
        schema   = item.get("schema", f"Database: {item.get('db_id', 'unknown')}"),
        question = item.get("question", ""),
    )

    return {
        "prompt"        : formatted_prompt,
        "db_id"         : item.get("db_id", ""),
        "question"      : item.get("question", ""),
        "ground_truth"  : item.get("sql", ""),
        "schema"        : item.get("schema", ""),
    }


def check_sample_quality(item: dict) -> tuple:
    """
    Check if formatted sample meets quality standards

    Returns:
        (is_valid: bool, reason: str)
    """
    text = item.get("text", "")

    # Check all sections present
    required = [
        "### Instruction:",
        "### Schema:",
        "### Question:",
        "### SQL:",
    ]

    for section in required:
        if section not in text:
            return False, f"Missing section: {section}"

    # Check not too short
    if len(text) < 50:
        return False, "Text too short"

    # Check not too long (max 1500 chars for training efficiency)
    if len(text) > 1500:
        return False, "Text too long"

    # Check SQL is not empty
    sql_part = text.split("### SQL:")[-1].strip()
    if not sql_part:
        return False, "Empty SQL"

    # Check question is not empty
    question_part = text.split("### Question:")[-1]
    question_part = question_part.split("### SQL:")[0].strip()
    if not question_part:
        return False, "Empty question"

    return True, "OK"


# ─────────────────────────────────────────
# 5. SHOW FORMATTED EXAMPLE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("FORMATTED SAMPLE EXAMPLE")
print("=" * 60)

example_train = format_sample_train(cleaned_train[0])
print(f"\n📌 Training Format:")
print("─" * 40)
print(example_train["text"])
print("─" * 40)
print(f"Total chars : {len(example_train['text'])}")
print(f"Total words : {len(example_train['text'].split())}")

print(f"\n📌 Inference Format:")
print("─" * 40)
example_inf = format_sample_inference(cleaned_train[0])
print(example_inf["prompt"])
print("─" * 40)
print(f"Ground truth SQL: {example_inf['ground_truth']}")


# ─────────────────────────────────────────
# 6. FORMAT ALL TRAINING DATA
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("FORMATTING ALL TRAINING DATA")
print("=" * 60)

print(f"\n🔄 Processing {len(cleaned_train)} training samples...")

formatted_train = []
skipped_train   = []
skip_reasons    = {}

for i, item in enumerate(cleaned_train):

    # Progress update every 1000
    if i % 1000 == 0 and i > 0:
        print(f"   Processed {i}/{len(cleaned_train)}...")

    try:
        formatted = format_sample_train(item)
        is_valid, reason = check_sample_quality(formatted)

        if is_valid:
            formatted_train.append(formatted)
        else:
            skipped_train.append(item)
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    except Exception as e:
        skipped_train.append(item)
        skip_reasons[str(e)] = skip_reasons.get(str(e), 0) + 1

print(f"\n   ✅ Formatted : {len(formatted_train)}")
print(f"   ⏭️  Skipped   : {len(skipped_train)}")

if skip_reasons:
    print(f"\n   Skip Reasons:")
    for reason, count in skip_reasons.items():
        print(f"      {reason:<30} : {count}")


# ─────────────────────────────────────────
# 7. FORMAT ALL VALIDATION DATA
# ─────────────────────────────────────────

print(f"\n🔄 Processing {len(cleaned_val)} validation samples...")

formatted_val       = []
skipped_val         = []
skip_reasons_val    = {}

for item in cleaned_val:

    try:
        formatted = format_sample_train(item)
        is_valid, reason = check_sample_quality(formatted)

        if is_valid:
            formatted_val.append(formatted)
        else:
            skipped_val.append(item)
            skip_reasons_val[reason] = skip_reasons_val.get(reason, 0) + 1

    except Exception as e:
        skipped_val.append(item)
        skip_reasons_val[str(e)] = skip_reasons_val.get(str(e), 0) + 1

print(f"\n   ✅ Formatted : {len(formatted_val)}")
print(f"   ⏭️  Skipped   : {len(skipped_val)}")


# ─────────────────────────────────────────
# 8. ANALYZE FORMATTED DATA
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("FORMATTED DATA ANALYSIS")
print("=" * 60)

# Text length analysis
text_lengths = [len(item['text']) for item in formatted_train]

print(f"\n📊 Text Length (characters):")
print(f"   Minimum : {min(text_lengths)}")
print(f"   Maximum : {max(text_lengths)}")
print(f"   Average : {sum(text_lengths)//len(text_lengths)}")

# Length distribution
under_300  = sum(1 for l in text_lengths if l < 300)
under_500  = sum(1 for l in text_lengths if 300 <= l < 500)
under_800  = sum(1 for l in text_lengths if 500 <= l < 800)
under_1200 = sum(1 for l in text_lengths if 800 <= l < 1200)
over_1200  = sum(1 for l in text_lengths if l >= 1200)
total      = len(text_lengths)

print(f"\n📊 Length Distribution:")
print(f"   Under 300  chars : {under_300:>5} ({under_300/total*100:.1f}%)")
print(f"   300 - 500  chars : {under_500:>5} ({under_500/total*100:.1f}%)")
print(f"   500 - 800  chars : {under_800:>5} ({under_800/total*100:.1f}%)")
print(f"   800 - 1200 chars : {under_1200:>5} ({under_1200/total*100:.1f}%)")
print(f"   Over 1200  chars : {over_1200:>5} ({over_1200/total*100:.1f}%)")

# SQL complexity
complexity = {
    "Simple SELECT"    : 0,
    "With WHERE"       : 0,
    "With JOIN"        : 0,
    "With GROUP BY"    : 0,
    "With HAVING"      : 0,
    "With ORDER BY"    : 0,
    "With Aggregation" : 0,
    "With Subquery"    : 0,
}

for item in formatted_train:
    sql = item['sql'].upper()
    if "WHERE"           in sql: complexity["With WHERE"]       += 1
    if "JOIN"            in sql: complexity["With JOIN"]        += 1
    if "GROUP BY"        in sql: complexity["With GROUP BY"]    += 1
    if "HAVING"          in sql: complexity["With HAVING"]      += 1
    if "ORDER BY"        in sql: complexity["With ORDER BY"]    += 1
    if any(f in sql for f in ["COUNT", "SUM", "AVG", "MAX", "MIN"]):
        complexity["With Aggregation"] += 1
    if sql.count("SELECT") > 1:
        complexity["With Subquery"]    += 1
    if (
        "WHERE"  not in sql and
        "JOIN"   not in sql and
        "GROUP"  not in sql
    ):
        complexity["Simple SELECT"]    += 1

print(f"\n📊 SQL Complexity Distribution:")
print(f"\n{'Type':<20} {'Count':<8} {'Pct':<8} Bar")
print("-" * 55)
for comp_type, count in complexity.items():
    pct = (count / total) * 100
    bar = "█" * int(pct / 5)
    print(f"{comp_type:<20} {count:<8} {pct:.1f}%   {bar}")

# Database distribution
db_counts = {}
for item in formatted_train:
    db = item['db_id']
    db_counts[db] = db_counts.get(db, 0) + 1

print(f"\n📊 Database Coverage:")
print(f"   Unique databases : {len(db_counts)}")
print(f"   Avg samples/db   : {total//len(db_counts)}")


# ─────────────────────────────────────────
# 9. SAVE FORMATTED DATA
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SAVING FORMATTED DATA")
print("=" * 60)

# Save as JSON (intermediate format)
with open("data/processed/formatted_train.json", "w",
          encoding="utf-8") as f:
    json.dump(formatted_train, f, indent=2, ensure_ascii=False)

with open("data/processed/formatted_val.json", "w",
          encoding="utf-8") as f:
    json.dump(formatted_val, f, indent=2, ensure_ascii=False)

print(f"\n✅ formatted_train.json : {len(formatted_train)} samples")
print(f"✅ formatted_val.json   : {len(formatted_val)} samples")


# ─────────────────────────────────────────
# 10. SAVE FORMAT REPORT
# ─────────────────────────────────────────

format_report = {
    "formatted_train"    : len(formatted_train),
    "formatted_val"      : len(formatted_val),
    "skipped_train"      : len(skipped_train),
    "skipped_val"        : len(skipped_val),
    "skip_reasons_train" : skip_reasons,
    "skip_reasons_val"   : skip_reasons_val,
    "text_length_stats"  : {
        "min" : min(text_lengths),
        "max" : max(text_lengths),
        "avg" : sum(text_lengths) // len(text_lengths),
    },
    "sql_complexity"     : complexity,
    "unique_databases"   : len(db_counts),
    "template_version"   : "v1",
    "ready_for_split"    : True,
}

with open("data/processed/format_report.json", "w") as f:
    json.dump(format_report, f, indent=2)

print(f"✅ format_report.json saved")


# ─────────────────────────────────────────
# 11. FINAL SUMMARY
# ─────────────────────────────────────────

print(f"\n{'='*60}")
print(f"FORMATTING COMPLETE!")
print(f"{'='*60}")
print(f"\n✅ Training samples formatted : {len(formatted_train)}")
print(f"✅ Validation samples formatted: {len(formatted_val)}")
print(f"✅ Template version            : v1")
print(f"✅ All files saved to          : data/processed/")
print(f"\n→ Next: Run data/split_data.py")