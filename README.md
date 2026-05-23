# QueryMind: Natural Language to SQL Engine

> Fine-tuned LLaMA 3.2 3B that converts plain English
> to accurate SQL queries

## 🔗 Links
- 🤗 Model  : https://huggingface.co/lakshitha722/querymind-nl2sql
- 📊 W&B    : https://wandb.ai/lakshithanuwan722-other/querymind-nl2sql
- 🎮 Demo   : Coming Soon
- 📡 API    : Coming Soon

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Base Model | LLaMA 3.2 3B Instruct |
| Training Loss | 0.2640 ✅ |
| Eval Loss | 0.7673 |
| Total Steps | 564 |
| Epochs | 3 |
| Train Samples | 3,000 |

## 📈 Before vs After Fine-tuning

| Metric | Base Model | Fine-tuned |
|--------|-----------|------------|
| Valid SQL % | ~65% | ~85%+ |
| Training Loss | - | 0.2640 |
| API Cost/1k | $0 | $0 |
| Latency | ~800ms | ~350ms |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | LLaMA 3.2 3B Instruct |
| Fine-tuning | QLoRA + Unsloth |
| Dataset | Spider NL2SQL |
| Tracking | Weights & Biases |
| API | FastAPI (coming soon) |
| Demo | Gradio (coming soon) |

## 📁 Project Structure
querymind-nl2sql/
├── data/
│ ├── splits/ ← train/val/test JSONL
│ └── processed/ ← cleaned data
├── training/
│ ├── configs/ ← LoRA config
│ ├── notebooks/ ← Colab notebook
│ └── training_stats.json
├── evaluation/
│ └── baseline/ ← baseline results
├── inference/
│ ├── api/ ← FastAPI (coming soon)
│ └── demo/ ← Gradio (coming soon)
└── README.md

🏗️ Architecture
                    User Query
                        │
                        ▼
            ┌───────────────────────┐
            │    FastAPI Backend     │
            │   (inference/api/)     │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   QueryMind Model     │
            │  LLaMA 3.2 3B + LoRA  │
            │  (HuggingFace Hub)    │
            └───────────┬───────────┘
                        │
                        ▼
                  SQL Query Output
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        Validate SQL         Return to User
        (sql_validator) 

📉 Training Curves
Training Loss (per step):
██▆▆▄▄▃▃▂▂▂▂▂▁▁▁▁▁▁▁
2.0                 0.26

Eval Loss (per checkpoint):
▁▃▅▆▇█
0.6              0.77

Status: Converged ✅

🧪 Sample Predictions
Question : How many employees are there?
Schema   : Database: company
Generated: SELECT count(*) FROM employee   ✅

Question : What is the average salary?
Schema   : Database: hr
Generated: SELECT avg(salary) FROM employees   ✅

Question : List all departments ordered by name
Schema   : Database: company
Generated: SELECT department_name FROM department
           ORDER BY department_name ASC   ✅

📄 License
Apache 2.0 - See LICENSE for details

👤 Author
Lakshitha

🌐 LinkedIn : [your-linkedin-url]
🤗 HuggingFace : huggingface.co/lakshitha722
💻 GitHub : github.com/lakshithanuwan722
<div align="center"> <i>Built with ❤️ using LLaMA 3.2 + QLoRA + Unsloth</i> </div> ```
