"""
Download Spider Schema using HuggingFace datasets API
"""

import os
import json
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("DOWNLOADING SPIDER SCHEMA")
print("=" * 60)


# ─────────────────────────────────────────
# 1. LOAD SPIDER DATASET WITH SCHEMA
# ─────────────────────────────────────────

print("\n📥 Loading Spider dataset...")
dataset = load_dataset("spider")

train_data = dataset["train"]
val_data   = dataset["validation"]

print(f"✅ Dataset loaded")
print(f"   Train : {len(train_data)}")
print(f"   Val   : {len(val_data)}")


# ─────────────────────────────────────────
# 2. PRINT ALL FIELDS IN DETAIL
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("ALL FIELDS IN SPIDER DATASET")
print("=" * 60)

sample = train_data[0]
print(f"\ndb_id    : {sample['db_id']}")
print(f"question : {sample['question']}")
print(f"query    : {sample['query']}")

# Print ALL fields with full values
print(f"\n--- All Fields ---")
for key, value in sample.items():
    print(f"\nKey: {key}")
    print(f"Value: {value}")


# ─────────────────────────────────────────
# 3. TRY DIFFERENT SPIDER DATASET VERSIONS
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("TRYING SPIDER WITH TABLES CONFIG")
print("=" * 60)

try:
    # Try loading with different config
    dataset_with_tables = load_dataset(
        "spider",
        trust_remote_code=True
    )
    print(f"\nConfigs available:")
    print(dataset_with_tables)
    
except Exception as e:
    print(f"Error: {e}")


# ─────────────────────────────────────────
# 4. BUILD SCHEMA FROM QUERY ITSELF
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("BUILDING SCHEMA FROM SQL QUERIES")
print("=" * 60)

# Since we cannot get full schema easily
# Extract table names directly from SQL queries
# This is good enough for fine-tuning

import re

def extract_tables_from_sql(sql: str) -> list:
    """Extract table names from SQL query"""
    sql_upper = sql.upper()
    tables    = []
    
    # Pattern: FROM table_name
    from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    from_matches = re.findall(from_pattern, sql, re.IGNORECASE)
    tables.extend(from_matches)
    
    # Pattern: JOIN table_name
    join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
    tables.extend(join_matches)
    
    # Remove duplicates
    tables = list(set(tables))
    return tables


def extract_columns_from_sql(sql: str) -> list:
    """Extract column references from SQL"""
    columns = []
    
    # Get everything between SELECT and FROM
    select_match = re.search(
        r'SELECT\s+(.*?)\s+FROM',
        sql,
        re.IGNORECASE | re.DOTALL
    )
    
    if select_match:
        select_part = select_match.group(1)
        
        # Split by comma
        parts = select_part.split(",")
        for part in parts:
            part = part.strip()
            # Remove function wrappers like COUNT(), AVG()
            part = re.sub(
                r'(COUNT|SUM|AVG|MAX|MIN|DISTINCT)\s*\(',
                '',
                part,
                flags=re.IGNORECASE
            )
            part = part.replace(")", "").replace("(", "").strip()
            
            if part and part != "*":
                # Handle table.column format
                if "." in part:
                    part = part.split(".")[-1]
                columns.append(part.strip())
    
    return columns


# ─────────────────────────────────────────
# 5. BUILD DB SCHEMA FROM ALL SAMPLES
# ─────────────────────────────────────────

print("\n🔄 Building schema lookup from SQL queries...")

# Collect all table/column info per database
db_schema_info = {}

all_data = list(train_data) + list(val_data)

for sample in all_data:
    db_id = sample['db_id']
    sql   = sample['query']
    
    if db_id not in db_schema_info:
        db_schema_info[db_id] = {
            "tables"  : set(),
            "columns" : set(),
        }
    
    # Extract tables and columns
    tables  = extract_tables_from_sql(sql)
    columns = extract_columns_from_sql(sql)
    
    db_schema_info[db_id]["tables"].update(tables)
    db_schema_info[db_id]["columns"].update(columns)


# Convert sets to lists for JSON
for db_id in db_schema_info:
    db_schema_info[db_id]["tables"]  = list(
        db_schema_info[db_id]["tables"]
    )
    db_schema_info[db_id]["columns"] = list(
        db_schema_info[db_id]["columns"]
    )


print(f"✅ Collected schema for {len(db_schema_info)} databases")


# ─────────────────────────────────────────
# 6. BUILD SCHEMA STRING FOR EACH DB
# ─────────────────────────────────────────

print("\n🔄 Building schema strings...")

schema_lookup = {}

for db_id, info in db_schema_info.items():
    tables  = info["tables"]
    columns = info["columns"]
    
    if not tables:
        schema_lookup[db_id] = f"Database: {db_id}"
        continue
    
    # Build schema string
    parts = []
    parts.append(f"Database: {db_id}")
    
    if tables:
        tables_str = ", ".join(sorted(tables))
        parts.append(f"Tables: {tables_str}")
    
    if columns:
        # Only show non-trivial columns
        clean_cols = [
            c for c in columns
            if c and len(c) > 1 and c.lower() not in
            ['*', 'null', 'true', 'false', '1', '0']
        ]
        if clean_cols:
            cols_str = ", ".join(sorted(set(clean_cols))[:20])
            parts.append(f"Columns include: {cols_str}")
    
    schema_lookup[db_id] = "\n".join(parts)


# ─────────────────────────────────────────
# 7. SHOW EXAMPLES
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("SCHEMA EXAMPLES")
print("=" * 60)

example_dbs = list(schema_lookup.keys())[:5]

for db_id in example_dbs:
    print(f"\n--- {db_id} ---")
    print(schema_lookup[db_id])


# ─────────────────────────────────────────
# 8. SAVE SCHEMA LOOKUP
# ─────────────────────────────────────────

with open("data/raw/schema_lookup.json", "w") as f:
    json.dump(schema_lookup, f, indent=2)

print(f"\n✅ Schema lookup saved!")
print(f"   File     : data/raw/schema_lookup.json")
print(f"   Databases: {len(schema_lookup)}")


# ─────────────────────────────────────────
# 9. ALSO SAVE RAW DB INFO
# ─────────────────────────────────────────

# Convert to serializable format
db_schema_serializable = {}
for db_id, info in db_schema_info.items():
    db_schema_serializable[db_id] = {
        "tables"  : info["tables"],
        "columns" : info["columns"],
    }

with open("data/raw/db_schema_info.json", "w") as f:
    json.dump(db_schema_serializable, f, indent=2)

print(f"✅ Raw DB info saved!")
print(f"   File     : data/raw/db_schema_info.json")

print(f"\n{'='*60}")
print(f"SCHEMA DOWNLOAD COMPLETE!")
print(f"{'='*60}")
print(f"\n→ Next: Run data/clean_data.py")