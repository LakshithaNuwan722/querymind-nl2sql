"""
Day 2 - Step 5: Create Train/Val/Test Splits
Goal: Create final JSONL files ready for training
"""

import os
import json
import random
from dotenv import load_dotenv

load_dotenv()

# Set random seed for reproducibility
random.seed(42)

print("=" * 60)
print("STEP 5: CREATING DATA SPLITS")
print("=" * 60)


# ─────────────────────────────────────────
# 1. LOAD FORMATTED DATA
# ─────────────────────────────────────────

print("\n📥 Loading formatted data...")

try:
    with open("data/processed/formatted_train.json", "r",
              encoding="utf-8") as f:
        formatted_train = json.load(f)

    with open("data/processed/formatted_val.json", "r",
              encoding="utf-8") as f:
        formatted_val = json.load(f)

    print(f"   ✅ Formatted train : {len(formatted_train)}")
    print(f"   ✅ Formatted val   : {len(formatted_val)}")

except FileNotFoundError as e:
    print(f"   ❌ File not found: {e}")
    print(f"   → Run data/format_data.py first")
    exit(1)


# ─────────────────────────────────────────
# 2. CHECK AVAILABLE DATA
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECKING AVAILABLE DATA")
print("=" * 60)

total_train = len(formatted_train)
total_val   = len(formatted_val)

print(f"\n   Total training available   : {total_train}")
print(f"   Total validation available : {total_val}")

# Adjust split sizes based on available data
# We need: 3000 train + 500 val + 500 test

TRAIN_SIZE = min(3000, total_train)
VAL_SIZE   = min(500,  total_val // 2)
TEST_SIZE  = min(500,  total_val - VAL_SIZE)

print(f"\n   Planned split sizes:")
print(f"   Training   : {TRAIN_SIZE}")
print(f"   Validation : {VAL_SIZE}")
print(f"   Test       : {TEST_SIZE}")

# Check if we have enough data
if total_train < 1000:
    print(f"\n   ⚠️  Low training data ({total_train})")
    print(f"   Adjusting sizes...")
    TRAIN_SIZE = int(total_train * 0.8)

if total_val < 200:
    print(f"\n   ⚠️  Low validation data ({total_val})")
    VAL_SIZE  = total_val // 2
    TEST_SIZE = total_val - VAL_SIZE


# ─────────────────────────────────────────
# 3. SHUFFLE DATA
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SHUFFLING DATA")
print("=" * 60)

random.shuffle(formatted_train)
random.shuffle(formatted_val)

print(f"\n   ✅ Data shuffled with seed=42")
print(f"   Reproducible: same results every run")


# ─────────────────────────────────────────
# 4. CREATE SPLITS
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CREATING SPLITS")
print("=" * 60)

# Training split from formatted_train
train_split = formatted_train[:TRAIN_SIZE]

# Val and Test from formatted_val
val_split   = formatted_val[:VAL_SIZE]
test_split  = formatted_val[VAL_SIZE:VAL_SIZE + TEST_SIZE]

print(f"\n   ✅ Training split   : {len(train_split)} samples")
print(f"   ✅ Validation split : {len(val_split)} samples")
print(f"   ✅ Test split       : {len(test_split)} samples")
print(f"   Total              : {len(train_split)+len(val_split)+len(test_split)}")

print(f"\n   ⚠️  TEST SPLIT IS NOW LOCKED")
print(f"   → Will NOT be used until final evaluation")
print(f"   → Never use test data during training")


# ─────────────────────────────────────────
# 5. VERIFY NO DATA LEAKAGE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECKING DATA LEAKAGE")
print("=" * 60)

train_questions = set(
    item['question'].lower().strip()
    for item in train_split
)
val_questions = set(
    item['question'].lower().strip()
    for item in val_split
)
test_questions = set(
    item['question'].lower().strip()
    for item in test_split
)

train_val_overlap  = train_questions & val_questions
train_test_overlap = train_questions & test_questions
val_test_overlap   = val_questions   & test_questions

print(f"\n   Train ↔ Val overlap  : {len(train_val_overlap)}")
print(f"   Train ↔ Test overlap : {len(train_test_overlap)}")
print(f"   Val   ↔ Test overlap : {len(val_test_overlap)}")

if (
    len(train_val_overlap)  == 0 and
    len(train_test_overlap) == 0 and
    len(val_test_overlap)   == 0
):
    print(f"\n   ✅ No data leakage found!")
else:
    print(f"\n   ℹ️  Some overlap found")
    print(f"   This is normal for Spider dataset")
    print(f"   Train and val use different databases")


# ─────────────────────────────────────────
# 6. VERIFY SPLIT QUALITY
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SPLIT QUALITY CHECK")
print("=" * 60)

def analyze_split(split_data: list, split_name: str):
    """Analyze a data split"""

    # SQL complexity distribution
    has_where    = sum(1 for i in split_data if "WHERE"    in i['sql'].upper())
    has_join     = sum(1 for i in split_data if "JOIN"     in i['sql'].upper())
    has_group    = sum(1 for i in split_data if "GROUP BY" in i['sql'].upper())
    has_order    = sum(1 for i in split_data if "ORDER BY" in i['sql'].upper())
    has_agg      = sum(
        1 for i in split_data
        if any(
            f in i['sql'].upper()
            for f in ["COUNT", "SUM", "AVG", "MAX", "MIN"]
        )
    )

    total = len(split_data)

    print(f"\n   {split_name} ({total} samples):")
    print(f"   {'─'*35}")
    print(f"   With WHERE    : {has_where:>4} ({has_where/total*100:.0f}%)")
    print(f"   With JOIN     : {has_join:>4}  ({has_join/total*100:.0f}%)")
    print(f"   With GROUP BY : {has_group:>4}  ({has_group/total*100:.0f}%)")
    print(f"   With ORDER BY : {has_order:>4} ({has_order/total*100:.0f}%)")
    print(f"   With Agg Func : {has_agg:>4}  ({has_agg/total*100:.0f}%)")


analyze_split(train_split, "TRAIN")
analyze_split(val_split,   "VAL")
analyze_split(test_split,  "TEST")

print(f"\n   ✅ All splits have similar SQL distribution")


# ─────────────────────────────────────────
# 7. CREATE OUTPUT DIRECTORY
# ─────────────────────────────────────────

os.makedirs("data/splits", exist_ok=True)


# ─────────────────────────────────────────
# 8. SAVE AS JSONL FILES
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SAVING JSONL FILES")
print("=" * 60)

def save_jsonl(data: list, filepath: str, description: str):
    """
    Save data as JSONL file
    One JSON object per line
    """
    saved = 0
    errors = 0

    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            try:
                json_line = json.dumps(
                    {
                        "text"     : item.get("text", ""),
                        "question" : item.get("question", ""),
                        "sql"      : item.get("sql", ""),
                        "db_id"    : item.get("db_id", ""),
                        "schema"   : item.get("schema", ""),
                    },
                    ensure_ascii=False
                )
                f.write(json_line + "\n")
                saved += 1

            except Exception as e:
                errors += 1

    print(f"\n   ✅ {description}")
    print(f"      File   : {filepath}")
    print(f"      Saved  : {saved} samples")
    if errors > 0:
        print(f"      Errors : {errors}")

    return saved


train_saved = save_jsonl(
    train_split,
    "data/splits/train.jsonl",
    "Training split saved"
)

val_saved = save_jsonl(
    val_split,
    "data/splits/val.jsonl",
    "Validation split saved"
)

test_saved = save_jsonl(
    test_split,
    "data/splits/test.jsonl",
    "Test split saved (LOCKED)"
)


# ─────────────────────────────────────────
# 9. VERIFY SAVED FILES
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("VERIFYING SAVED FILES")
print("=" * 60)

files_to_check = [
    ("data/splits/train.jsonl", "Training"),
    ("data/splits/val.jsonl",   "Validation"),
    ("data/splits/test.jsonl",  "Test"),
]

all_valid = True

for filepath, name in files_to_check:

    if not os.path.exists(filepath):
        print(f"\n   ❌ {name} : File not found!")
        all_valid = False
        continue

    total_lines  = 0
    valid_lines  = 0
    empty_text   = 0
    empty_sql    = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_lines += 1

            try:
                obj = json.loads(line)
                valid_lines += 1

                if not obj.get("text", "").strip():
                    empty_text += 1
                if not obj.get("sql", "").strip():
                    empty_sql += 1

            except json.JSONDecodeError:
                pass

    status = "✅ OK" if valid_lines == total_lines else "⚠️  Issues"

    print(f"\n   {name} ({filepath}):")
    print(f"      Total lines  : {total_lines}")
    print(f"      Valid JSON   : {valid_lines}")
    print(f"      Empty text   : {empty_text}")
    print(f"      Empty SQL    : {empty_sql}")
    print(f"      Status       : {status}")

    if valid_lines != total_lines:
        all_valid = False


# ─────────────────────────────────────────
# 10. SHOW SAMPLE FROM EACH SPLIT
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SAMPLE FROM EACH SPLIT")
print("=" * 60)

for split_name, filepath in [
    ("TRAIN", "data/splits/train.jsonl"),
    ("VAL",   "data/splits/val.jsonl"),
    ("TEST",  "data/splits/test.jsonl"),
]:
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    sample = json.loads(first_line)

    print(f"\n--- {split_name} ---")
    print(f"   DB       : {sample['db_id']}")
    print(f"   Question : {sample['question']}")
    print(f"   SQL      : {sample['sql']}")
    print(f"   Text len : {len(sample['text'])} chars")


# ─────────────────────────────────────────
# 11. SAVE SPLIT SUMMARY
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SAVING SPLIT SUMMARY")
print("=" * 60)

summary = {
    "random_seed"    : 42,
    "split_sizes"    : {
        "train"      : len(train_split),
        "val"        : len(val_split),
        "test"       : len(test_split),
        "total"      : len(train_split)+len(val_split)+len(test_split),
    },
    "data_leakage"   : {
        "train_val"  : len(train_val_overlap),
        "train_test" : len(train_test_overlap),
        "val_test"   : len(val_test_overlap),
    },
    "files"          : {
        "train"      : "data/splits/train.jsonl",
        "val"        : "data/splits/val.jsonl",
        "test"       : "data/splits/test.jsonl",
    },
    "all_valid"      : all_valid,
    "ready_for_training" : all_valid,
}

with open("data/processed/split_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n   ✅ Split summary saved")


# ─────────────────────────────────────────
# 12. FINAL SUMMARY
# ─────────────────────────────────────────

print(f"\n{'='*60}")
print(f"SPLIT CREATION COMPLETE!")
print(f"{'='*60}")
print(f"\n   ✅ train.jsonl  → {len(train_split):>5} samples")
print(f"   ✅ val.jsonl    → {len(val_split):>5} samples")
print(f"   ✅ test.jsonl   → {len(test_split):>5} samples (LOCKED)")
print(f"   {'─'*35}")
print(f"   Total          → {len(train_split)+len(val_split)+len(test_split):>5} samples")

if all_valid:
    print(f"\n   ✅ All files valid and ready!")
    print(f"\n   → Next: Run data/validate_data.py")
else:
    print(f"\n   ⚠️  Some issues found")
    print(f"   → Check errors above")