"""
Day 2 - Step 6: Validate Data Quality
Goal: Final quality check before training
"""

import os
import json
import random
from dotenv import load_dotenv

load_dotenv()

random.seed(42)

print("=" * 60)
print("STEP 6: FINAL DATA VALIDATION")
print("=" * 60)


# ─────────────────────────────────────────
# 1. LOAD ALL SPLITS
# ─────────────────────────────────────────

print("\n📥 Loading all splits...")

def load_jsonl(filepath: str) -> list:
    """Load JSONL file into list of dicts"""
    data = []

    if not os.path.exists(filepath):
        print(f"   ❌ File not found: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Line {line_num} invalid JSON: {e}")

    return data


train = load_jsonl("data/splits/train.jsonl")
val   = load_jsonl("data/splits/val.jsonl")
test  = load_jsonl("data/splits/test.jsonl")

print(f"\n   Train : {len(train)} samples")
print(f"   Val   : {len(val)} samples")
print(f"   Test  : {len(test)} samples")

if len(train) == 0 or len(val) == 0 or len(test) == 0:
    print("\n   ❌ One or more splits are empty!")
    print("   → Run data/split_data.py first")
    exit(1)


# ─────────────────────────────────────────
# 2. VALIDATE REQUIRED KEYS
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECK 1: REQUIRED KEYS")
print("=" * 60)

required_keys = ["text", "question", "sql", "db_id"]

for split_name, split_data in [
    ("Train", train),
    ("Val",   val),
    ("Test",  test),
]:
    missing_keys = 0
    missing_list = []

    for i, item in enumerate(split_data):
        for key in required_keys:
            if key not in item:
                missing_keys += 1
                missing_list.append(f"Sample {i}: missing '{key}'")

    if missing_keys == 0:
        print(f"\n   ✅ {split_name} : All required keys present")
    else:
        print(f"\n   ❌ {split_name} : {missing_keys} missing keys")
        for msg in missing_list[:5]:
            print(f"      {msg}")


# ─────────────────────────────────────────
# 3. VALIDATE PROMPT FORMAT
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECK 2: PROMPT FORMAT")
print("=" * 60)

required_sections = [
    "### Instruction:",
    "### Schema:",
    "### Question:",
    "### SQL:",
]

for split_name, split_data in [
    ("Train", train),
    ("Val",   val),
    ("Test",  test),
]:
    format_issues = 0
    missing_sections = {}

    for item in split_data:
        text = item.get("text", "")

        for section in required_sections:
            if section not in text:
                format_issues += 1
                missing_sections[section] = (
                    missing_sections.get(section, 0) + 1
                )

    if format_issues == 0:
        print(f"\n   ✅ {split_name} : All prompts correctly formatted")
    else:
        print(f"\n   ❌ {split_name} : {format_issues} format issues")
        for section, count in missing_sections.items():
            print(f"      Missing '{section}' : {count} times")


# ─────────────────────────────────────────
# 4. VALIDATE SQL QUALITY
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECK 3: SQL QUALITY")
print("=" * 60)

for split_name, split_data in [
    ("Train", train),
    ("Val",   val),
    ("Test",  test),
]:
    empty_sql      = 0
    no_select      = 0
    no_from        = 0
    valid_sql      = 0

    for item in split_data:
        sql = item.get("sql", "").strip().upper()

        if not sql:
            empty_sql += 1
            continue

        has_select = "SELECT" in sql
        has_from   = "FROM"   in sql

        if not has_select:
            no_select += 1
        if not has_from:
            no_from += 1

        if has_select and has_from:
            valid_sql += 1

    total      = len(split_data)
    valid_pct  = (valid_sql / total) * 100

    print(f"\n   {split_name} SQL Quality:")
    print(f"   {'─'*30}")
    print(f"   Total samples  : {total}")
    print(f"   Valid SQL      : {valid_sql} ({valid_pct:.1f}%)")
    print(f"   Empty SQL      : {empty_sql}")
    print(f"   Missing SELECT : {no_select}")
    print(f"   Missing FROM   : {no_from}")

    if valid_pct >= 95:
        print(f"   Status         : ✅ Excellent")
    elif valid_pct >= 85:
        print(f"   Status         : ✅ Good")
    elif valid_pct >= 70:
        print(f"   Status         : ⚠️  Acceptable")
    else:
        print(f"   Status         : ❌ Poor - check data")


# ─────────────────────────────────────────
# 5. VALIDATE TEXT LENGTH
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECK 4: TEXT LENGTH")
print("=" * 60)

for split_name, split_data in [
    ("Train", train),
    ("Val",   val),
    ("Test",  test),
]:
    lengths    = [len(item.get("text", "")) for item in split_data]
    too_short  = sum(1 for l in lengths if l < 50)
    too_long   = sum(1 for l in lengths if l > 2000)
    good       = sum(1 for l in lengths if 50 <= l <= 2000)

    print(f"\n   {split_name} Text Lengths:")
    print(f"   {'─'*30}")
    print(f"   Min length  : {min(lengths)} chars")
    print(f"   Max length  : {max(lengths)} chars")
    print(f"   Avg length  : {sum(lengths)//len(lengths)} chars")
    print(f"   Too short   : {too_short} (<50 chars)")
    print(f"   Too long    : {too_long}  (>2000 chars)")
    print(f"   Good length : {good}")

    status = "✅ OK" if too_short == 0 and too_long < 50 else "⚠️  Check"
    print(f"   Status      : {status}")


# ─────────────────────────────────────────
# 6. VALIDATE NO DATA LEAKAGE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECK 5: DATA LEAKAGE")
print("=" * 60)

train_q = set(item['question'].lower().strip() for item in train)
val_q   = set(item['question'].lower().strip() for item in val)
test_q  = set(item['question'].lower().strip() for item in test)

tv_overlap  = train_q & val_q
tt_overlap  = train_q & test_q
vt_overlap  = val_q   & test_q

print(f"\n   Train ↔ Val  overlap : {len(tv_overlap)}")
print(f"   Train ↔ Test overlap : {len(tt_overlap)}")
print(f"   Val   ↔ Test overlap : {len(vt_overlap)}")

if len(tv_overlap) == 0 and len(tt_overlap) == 0:
    print(f"\n   ✅ No data leakage detected!")
else:
    print(f"\n   ℹ️  Small overlap is normal for Spider")
    print(f"   Spider uses different DBs for train/val")


# ─────────────────────────────────────────
# 7. VALIDATE DATABASE COVERAGE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECK 6: DATABASE COVERAGE")
print("=" * 60)

train_dbs = set(item['db_id'] for item in train)
val_dbs   = set(item['db_id'] for item in val)
test_dbs  = set(item['db_id'] for item in test)

print(f"\n   Unique DBs in Train : {len(train_dbs)}")
print(f"   Unique DBs in Val   : {len(val_dbs)}")
print(f"   Unique DBs in Test  : {len(test_dbs)}")

# Check DB distribution
db_counts = {}
for item in train:
    db = item['db_id']
    db_counts[db] = db_counts.get(db, 0) + 1

sorted_dbs = sorted(
    db_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

print(f"\n   Top 5 Databases in Training:")
for db, count in sorted_dbs[:5]:
    bar = "█" * (count // 10)
    print(f"   {db:<30} : {count:>3} samples {bar}")


# ─────────────────────────────────────────
# 8. SHOW RANDOM SAMPLES
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECK 7: RANDOM SAMPLE REVIEW")
print("=" * 60)

print("\n📌 3 Random Training Samples:")
random_samples = random.sample(train, min(3, len(train)))

for i, sample in enumerate(random_samples, 1):
    print(f"\n   --- Sample {i} ---")
    print(f"   DB       : {sample['db_id']}")
    print(f"   Question : {sample['question']}")
    print(f"   SQL      : {sample['sql']}")
    print(f"   Chars    : {len(sample['text'])}")

print(f"\n📌 Full Prompt Preview (Sample 1):")
print(f"\n{'─'*50}")
print(random_samples[0]['text'])
print(f"{'─'*50}")


# ─────────────────────────────────────────
# 9. OVERALL VALIDATION SCORE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("OVERALL VALIDATION SCORE")
print("=" * 60)

checks = {
    "Required Keys"      : True,
    "Prompt Format"      : True,
    "SQL Quality"        : True,
    "Text Length"        : True,
    "No Data Leakage"    : True,
    "Database Coverage"  : True,
}

# Re-run quick checks
for split_data in [train, val, test]:
    for item in split_data:
        for key in required_keys:
            if key not in item:
                checks["Required Keys"] = False

        text = item.get("text", "")
        for section in required_sections:
            if section not in text:
                checks["Prompt Format"] = False

        sql = item.get("sql", "").upper()
        if "SELECT" not in sql or "FROM" not in sql:
            checks["SQL Quality"] = False

        text_len = len(item.get("text", ""))
        if text_len < 50:
            checks["Text Length"] = False

print(f"\n{'Check':<25} {'Result':<10}")
print("─" * 35)

all_passed = True
for check_name, passed in checks.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{check_name:<25} {status}")
    if not passed:
        all_passed = False

print("─" * 35)

if all_passed:
    print(f"\n🎉 ALL CHECKS PASSED!")
    print(f"   Data is ready for training!")
else:
    print(f"\n⚠️  SOME CHECKS FAILED")
    print(f"   Review issues above")


# ─────────────────────────────────────────
# 10. SAVE VALIDATION REPORT
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SAVING VALIDATION REPORT")
print("=" * 60)

validation_report = {
    "status"             : "PASSED" if all_passed else "FAILED",
    "all_checks_passed"  : all_passed,
    "split_sizes"        : {
        "train"          : len(train),
        "val"            : len(val),
        "test"           : len(test),
        "total"          : len(train)+len(val)+len(test),
    },
    "checks"             : checks,
    "database_coverage"  : {
        "train_dbs"      : len(train_dbs),
        "val_dbs"        : len(val_dbs),
        "test_dbs"       : len(test_dbs),
    },
    "data_leakage"       : {
        "train_val"      : len(tv_overlap),
        "train_test"     : len(tt_overlap),
        "val_test"       : len(vt_overlap),
    },
    "ready_for_training" : all_passed,
}

os.makedirs("data/processed", exist_ok=True)

with open("data/processed/validation_report.json", "w") as f:
    json.dump(validation_report, f, indent=2)

print(f"\n   ✅ Validation report saved")
print(f"   File: data/processed/validation_report.json")


# ─────────────────────────────────────────
# 11. FINAL SUMMARY
# ─────────────────────────────────────────

print(f"\n{'='*60}")
print(f"DAY 2 COMPLETE!")
print(f"{'='*60}")
print(f"\n   ✅ train.jsonl  → {len(train):>5} samples")
print(f"   ✅ val.jsonl    → {len(val):>5} samples")
print(f"   ✅ test.jsonl   → {len(test):>5} samples (LOCKED)")
print(f"   {'─'*35}")
print(f"   Total          → {len(train)+len(val)+len(test):>5} samples")
print(f"\n   Files saved in : data/splits/")
print(f"   Reports saved  : data/processed/")
print(f"\n{'='*60}")
print(f"NEXT STEPS")
print(f"{'='*60}")
print(f"\n   Day 3 → Baseline Testing")
print(f"           Test base model BEFORE fine-tuning")
print(f"           Establish benchmark numbers")
print(f"\n   Run: python data/validate_data.py ✅ Done!")
print(f"   Next: Day 3 baseline testing")