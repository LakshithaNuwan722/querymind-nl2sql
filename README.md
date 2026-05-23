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
│
├── 📁 data/
│   ├── 📁 raw/
│   │   ├── exploration_summary.json     ← dataset statistics
│   │   ├── schema_lookup.json           ← DB schema info
│   │   └── db_schema_info.json          ← raw schema data
│   │
│   ├── 📁 processed/
│   │   ├── cleaned_train.json           ← cleaned training data
│   │   ├── cleaned_val.json             ← cleaned validation data
│   │   ├── formatted_train.json         ← formatted for training
│   │   ├── formatted_val.json           ← formatted for validation
│   │   ├── prompt_templates.json        ← prompt templates v1
│   │   ├── cleaning_report.json         ← data cleaning report
│   │   ├── format_report.json           ← formatting report
│   │   ├── split_summary.json           ← split statistics
│   │   └── validation_report.json       ← quality validation
│   │
│   ├── 📁 splits/
│   │   ├── train.jsonl                  ← 3000 training samples
│   │   ├── val.jsonl                    ← 500 validation samples
│   │   └── test.jsonl                   ← 500 test samples (locked)
│   │
│   ├── explore_dataset.py               ← dataset exploration
│   ├── load_data.py                     ← data loading
│   ├── download_schema.py               ← schema downloader
│   ├── clean_data.py                    ← data cleaning
│   ├── format_data.py                   ← data formatting
│   ├── split_data.py                    ← train/val/test splits
│   └── validate_data.py                 ← quality validation
│
├── 📁 training/
│   ├── 📁 configs/
│   │   └── lora_config.json             ← LoRA hyperparameters
│   │
│   ├── 📁 notebooks/
│   │   └── querymind_finetuning.ipynb   ← Colab training notebook
│   │
│   ├── training_stats.json              ← final training results
│   └── TRAINING.md                      ← training documentation
│
├── 📁 evaluation/
│   ├── 📁 baseline/
│   │   ├── __init__.py
│   │   ├── config.py                    ← baseline configuration
│   │   ├── sql_validator.py             ← SQL validation logic
│   │   ├── run_baseline.py              ← baseline testing script
│   │   ├── baseline_llama_8b.json       ← Llama 8B results
│   │   ├── baseline_llama_70b.json      ← Llama 70B results
│   │   └── baseline_combined.json       ← combined comparison
│   │
│   ├── 📁 results/                      ← evaluation outputs
│   ├── 📁 charts/                       ← comparison charts
│   └── __init__.py
│
├── 📁 inference/
│   ├── 📁 api/                          ← FastAPI endpoint (coming)
│   └── 📁 demo/                         ← Gradio demo (coming)
│
├── 📁 monitoring/                        ← metrics + dashboards
│
├── 📄 .env                              ← API keys (not in git)
├── 📄 .gitignore                        ← ignored files
├── 📄 requirements.txt                  ← Python dependencies
└── 📄 README.md                         ← this file

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