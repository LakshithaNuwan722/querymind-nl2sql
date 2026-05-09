"""
SQL Validator
Goal: Check if generated SQL is valid and correct
"""

import re
import sqlite3


def is_valid_sql_syntax(sql: str) -> bool:
    """
    Check if SQL has valid basic syntax
    Does NOT execute, just checks structure
    """
    if not sql or not sql.strip():
        return False

    sql_clean = sql.strip().upper()

    # Remove SQL code block markers if present
    sql_clean = sql_clean.replace("```SQL", "").replace("```", "").strip()

    # Must start with SELECT
    if not sql_clean.startswith("SELECT"):
        return False

    # Must have FROM
    if "FROM" not in sql_clean:
        return False

    # Must be reasonable length
    if len(sql_clean) < 10:
        return False

    if len(sql_clean) > 1000:
        return False

    return True


def clean_sql_output(raw_output: str) -> str:
    """
    Clean LLM output to extract just the SQL
    LLMs sometimes add extra text, explanations, code blocks
    """
    if not raw_output:
        return ""

    # Remove markdown code blocks
    # Pattern: ```sql ... ``` or ``` ... ```
    cleaned = re.sub(
        r'```sql\s*', '', raw_output, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r'```\s*', '', cleaned
    )

    # Split by newline and take meaningful lines
    lines = cleaned.strip().split('\n')

    sql_lines = []
    for line in lines:
        line = line.strip()

        # Skip empty lines at start
        if not sql_lines and not line:
            continue

        # Skip explanation lines
        # These usually start with words like "This", "The", "Note"
        if sql_lines and re.match(
            r'^(This|The|Note|Here|Result|Output|Answer)',
            line, re.IGNORECASE
        ):
            break

        sql_lines.append(line)

    sql = ' '.join(sql_lines).strip()

    # Remove trailing semicolon
    sql = sql.rstrip(';').strip()

    # Remove leading/trailing quotes
    sql = sql.strip('"').strip("'")

    return sql


def calculate_sql_similarity(
    generated: str,
    reference: str
) -> float:
    """
    Calculate similarity between generated and reference SQL
    Simple keyword-based similarity
    Returns score between 0 and 1
    """
    if not generated or not reference:
        return 0.0

    gen_upper = generated.upper()
    ref_upper = reference.upper()

    # Extract key components
    keywords = [
        "SELECT", "FROM", "WHERE", "GROUP BY",
        "ORDER BY", "HAVING", "JOIN", "COUNT",
        "SUM", "AVG", "MAX", "MIN", "DISTINCT",
        "LIMIT", "AND", "OR", "NOT", "IN",
        "LIKE", "BETWEEN",
    ]

    gen_keywords = set(kw for kw in keywords if kw in gen_upper)
    ref_keywords = set(kw for kw in keywords if kw in ref_upper)

    if not ref_keywords:
        return 0.0

    # Jaccard similarity on keywords
    intersection = gen_keywords & ref_keywords
    union        = gen_keywords | ref_keywords

    keyword_sim = len(intersection) / len(union) if union else 0

    # Check table name match
    # Extract table names after FROM and JOIN
    def extract_tables(sql):
        tables = set()
        from_match = re.findall(
            r'FROM\s+(\w+)', sql, re.IGNORECASE
        )
        join_match = re.findall(
            r'JOIN\s+(\w+)', sql, re.IGNORECASE
        )
        tables.update(t.lower() for t in from_match)
        tables.update(t.lower() for t in join_match)
        return tables

    gen_tables = extract_tables(generated)
    ref_tables = extract_tables(reference)

    if ref_tables:
        table_match = len(
            gen_tables & ref_tables
        ) / len(ref_tables)
    else:
        table_match = 1.0

    # Combined score
    score = (keyword_sim * 0.5) + (table_match * 0.5)
    return round(score, 3)


def evaluate_single_prediction(
    generated_sql : str,
    reference_sql : str,
    question      : str,
) -> dict:
    """
    Evaluate a single SQL prediction

    Returns dict with all metrics
    """
    # Clean the generated SQL
    cleaned_sql = clean_sql_output(generated_sql)

    # Check syntax validity
    is_valid = is_valid_sql_syntax(cleaned_sql)

    # Calculate similarity
    similarity = calculate_sql_similarity(
        cleaned_sql,
        reference_sql
    )

    # Check exact match (case insensitive)
    exact_match = (
        cleaned_sql.upper().strip() ==
        reference_sql.upper().strip()
    )

    # Classify result
    if exact_match:
        result = "exact_match"
    elif is_valid and similarity >= 0.7:
        result = "high_similarity"
    elif is_valid and similarity >= 0.4:
        result = "medium_similarity"
    elif is_valid:
        result = "valid_but_different"
    else:
        result = "invalid_sql"

    return {
        "question"       : question,
        "generated_sql"  : cleaned_sql,
        "reference_sql"  : reference_sql,
        "is_valid"       : is_valid,
        "exact_match"    : exact_match,
        "similarity"     : similarity,
        "result"         : result,
    }


def summarize_results(predictions: list) -> dict:
    """
    Summarize evaluation results

    Args:
        predictions: list of evaluate_single_prediction outputs

    Returns:
        dict with summary metrics
    """
    total = len(predictions)

    if total == 0:
        return {}

    valid_count       = sum(1 for p in predictions if p['is_valid'])
    exact_match_count = sum(1 for p in predictions if p['exact_match'])
    similarities      = [p['similarity'] for p in predictions]

    high_sim   = sum(1 for p in predictions if p['similarity'] >= 0.7)
    medium_sim = sum(1 for p in predictions if 0.4 <= p['similarity'] < 0.7)
    low_sim    = sum(1 for p in predictions if p['similarity'] < 0.4)

    result_counts = {}
    for p in predictions:
        r = p['result']
        result_counts[r] = result_counts.get(r, 0) + 1

    return {
        "total_samples"      : total,
        "valid_sql_count"    : valid_count,
        "valid_sql_pct"      : round((valid_count / total) * 100, 1),
        "exact_match_count"  : exact_match_count,
        "exact_match_pct"    : round((exact_match_count / total) * 100, 1),
        "avg_similarity"     : round(sum(similarities) / total, 3),
        "high_similarity"    : high_sim,
        "medium_similarity"  : medium_sim,
        "low_similarity"     : low_sim,
        "result_breakdown"   : result_counts,
    }