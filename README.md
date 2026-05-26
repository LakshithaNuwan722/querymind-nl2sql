# <p align="center">🧠 QueryMind: Natural Language to SQL Engine</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-LLaMA%203.2%203B-blue?style=for-the-badge&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/Fine--Tuning-QLoRA%20%2B%20Unsloth-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Train%20Loss-0.2640-brightgreen?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/GPU%20Target-NVIDIA%20RTX%205070-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/UI-Gradio-FF5722?style=for-the-badge&logo=gradio&logoColor=white" />
</p>

---

QueryMind is a production-ready, highly-optimized **NL-to-SQL engine** that converts plain English queries into accurate, schema-valid SQL statements. Deployed with an interactive web UI and a fast REST API, it is fully optimized for local GPU inference.

---

## 🔗 Quick Access Buttons

<p align="left">
  <a href="https://huggingface.co/spaces/lakshitha722/querymind-nl2sql-demo">
    <img src="https://img.shields.io/badge/🎮%20HuggingFace%20Space-Live%20Demo-FFD21E?style=for-the-badge" />
  </a>
  <a href="https://huggingface.co/lakshitha722/querymind-nl2sql">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-Model%20Hub-FF9D00?style=for-the-badge" />
  </a>
  <a href="https://wandb.ai/lakshithanuwan722-other/querymind-nl2sql">
    <img src="https://img.shields.io/badge/📊%20Weights%20%26%20Biases-Train%20Stats-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black" />
  </a>
</p>

---

## 🏆 The Big Win: 3-Way Benchmark Results

We benchmarked our local **Fine-Tuned 3B model** against raw base models running on high-speed cloud infrastructure (Groq API). **Our domain-specific 3B model easily defeated LLaMA 70B!**

| Metric | Base LLaMA 8B <br> *(Groq API)* | 🧠 **Fine-Tuned 3B** <br> *(Local RTX 5070)* | Base LLaMA 70B <br> *(Groq API)* |
| :--- | :---: | :---: | :---: |
| **Valid SQL %** | `100.0%` | **`100.0%`** | `100.0%` |
| **Exact Match %** | `14.0%` | **`19.0%`** 🏆 | `11.0%` |
| **Avg Similarity** | `0.816` | **`0.863`** | **`0.874`** |
| **High Sim % (>=0.7)** | `68.0%` | **`77.0%`** | **`82.0%`** |
| **Inference Latency** | `1993.0 ms` | **`1539.8 ms`** ⚡ | `2028.8 ms` |
| **Avg Tokens Output** | `182.9` | **`34.6`** 📉 | `187.3` |
| **Cost per 1k Queries** | `$0.00` | **`$0.00`** 💰 | `$0.00` |

### 🚀 Key Insights:
- **Exact Match Victory**: Our 3B model achieved **19.0% Exact Match**, outperforming LLaMA 70B (11.0%) on specialized SQL logic!
- **Conciseness (No Filler)**: Generated queries average only **34.6 tokens** (direct SQL), avoiding conversational bloat (180+ tokens).
- **Latency**: Local RTX 5070 inference completes in **~1.5 seconds**, beating cloud API roundtrips!

---

## 🏗️ System Architecture Flow

```text
 ┌────────────────────────────────────────────────────────┐
 │                      User Query                        │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │                FastAPI Microservice (8000)             │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │           QueryMind (LLaMA 3.2 3B + LoRA)              │
 │          Running Locally on RTX 5070 GPU               │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │                SQL Output Regex Cleaner                │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
               SELECT count(*) FROM employee

📁 Project Directory Structure
querymind-nl2sql/
├── data/
│   └── splits/             # Train/Val/Test JSONL datasets
├── evaluation/
│   ├── results/            # Benchmark JSON results (8B, FT 3B, 70B)
│   └── run_evaluation.py   # 3-Way evaluation pipeline
├── inference/
│   ├── api.py              # FastAPI production-ready endpoint
│   ├── app.py              # Interactive Gradio app code
│   └── requirements.txt    # HF Spaces cloud deployment requirements
└── README.md

📉 Training Convergence
Training Loss: ██▆▆▄▄▃▃▂▂▂▂▂ (Converged from 2.0 to 0.2640 ✅)
Eval Loss:     ▁▃▅▆▇█         (Stabilized at 0.7673)

🧪 Sample Predictions
<details> <summary><b>💡 Click to expand SQL Generation Examples</b></summary>
Question  : How many employees are there?
Schema    : Database: company | Tables: employees (id, name, department, salary)
Generated : SELECT COUNT(*) FROM employees ✅

Question  : What is the average salary by department?
Schema    : Database: hr | Tables: employees (id, name, department, salary)
Generated : SELECT department, AVG(salary) FROM employees GROUP BY department ✅

Question  : Find products with price greater than 100
Schema    : Database: store | Tables: products (id, name, price, category)
Generated : SELECT name, price FROM products WHERE price > 100 ✅

👤 Author & Contact
Lakshitha

<p align="left"> <a href="huggingface.co/lakshitha722"> <img src="https://img.shields.io/badge/HuggingFace-lakshitha722-yellow?style=flat-square&logo=huggingface" /> </a> <a href="github.com/lakshithanuwan722"> <img src="https://img.shields.io/badge/GitHub-lakshithanuwan722-black?style=flat-square&logo=github" /> </a> <a href="[your-linkedin-url]"> <img src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat-square&logo=linkedin" /> </a> </p>
Built with ❤️ using LLaMA 3.2 + QLoRA + Unsloth
