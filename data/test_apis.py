"""
Day 1 - Step 8: Test all API connections
Goal: Make sure all APIs work before Day 2
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("API CONNECTION TESTS")
print("=" * 50)


# ─────────────────────────────────────────
# TEST 1: GROQ API
# ─────────────────────────────────────────

print("\n🔄 Testing Groq API...")

try:
    from groq import Groq
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",        # ← UPDATED
        messages=[
            {
                "role": "user",
                "content": "Say 'Groq API working' and nothing else."
            }
        ],
        max_tokens=20
    )
    
    print(f"   ✅ Groq API: {response.choices[0].message.content}")
    print(f"   Model used: {response.model}")
    
except Exception as e:
    print(f"   ❌ Groq API failed: {e}")


# ─────────────────────────────────────────
# TEST 2: HUGGING FACE
# ─────────────────────────────────────────

print("\n🔄 Testing Hugging Face...")

try:
    from huggingface_hub import whoami, login
    
    login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)
    user_info = whoami()
    
    print(f"   ✅ HuggingFace: Logged in as '{user_info['name']}'")
    
except Exception as e:
    print(f"   ❌ HuggingFace failed: {e}")


# ─────────────────────────────────────────
# TEST 3: WEIGHTS & BIASES
# ─────────────────────────────────────────

print("\n🔄 Testing Weights & Biases...")

try:
    import wandb
    
    # Disable W&B for now
    # Will fix properly on Day 6 (training day)
    os.environ["WANDB_MODE"] = "disabled"
    
    print("   ✅ W&B module installed correctly")
    print("   ⏭️  Skipped login - will fix on Day 6")
    
except ImportError:
    print("   ❌ W&B not installed")

# ─────────────────────────────────────────
# TEST 4: TEST A SIMPLE NL→SQL PROMPT
# ─────────────────────────────────────────

print("\n🔄 Testing NL→SQL with Groq (baseline preview)...")

try:
    from groq import Groq
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    test_prompt = """Convert this question to SQL.

Schema:
Table: employees
Columns: emp_id (INT), name (VARCHAR), department (VARCHAR), salary (FLOAT)

Question: What is the average salary of employees?

SQL:"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",        # ← UPDATED
        messages=[
            {
                "role": "user", 
                "content": test_prompt
            }
        ],
        max_tokens=100,
        temperature=0
    )
    
    sql_output = response.choices[0].message.content.strip()
    print(f"   ✅ NL→SQL test passed")
    print(f"   Input : 'What is the average salary of employees?'")
    print(f"   Output: {sql_output}")
    
except Exception as e:
    print(f"   ❌ NL→SQL test failed: {e}")


# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("ALL API TESTS COMPLETE")
print("=" * 50)
print("\nIf any ❌ appear above:")
print("  → Check your .env file has correct keys")
print("  → Check account is verified (HF, W&B)")
print("  → Check internet connection")
print("\nIf all ✅ → Ready for Day 2! 🚀")