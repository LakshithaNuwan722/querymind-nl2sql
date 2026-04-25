# QueryMind: Natural Language to SQL Engine

> Fine-tuned LLaMA 3.2 3B model that converts plain English 
> questions to accurate SQL queries

## 🚧 Status: In Development

## 📊 Results (Updated after training)

| Metric | Base LLaMA 3B | Fine-tuned | GPT-4 Mini |
|--------|--------------|------------|------------|
| Valid SQL % | TBD | TBD | TBD |
| Accuracy % | TBD | TBD | TBD |
| Latency (ms) | TBD | TBD | TBD |
| Cost/1k queries | $0 | $0 | $6.00 |

## 🛠️ Tech Stack
- **Model**: LLaMA 3.2 3B + QLoRA fine-tuning
- **Framework**: Unsloth + HuggingFace Transformers
- **Dataset**: Spider (NL2SQL benchmark)
- **Tracking**: Weights & Biases
- **API**: FastAPI
- **Demo**: Gradio on HF Spaces

## 📁 Project Structure
querymind-nl2sql/
├── data/ # Dataset preparation
├── training/ # Fine-tuning scripts
├── evaluation/ # Evaluation framework
├── inference/ # API + Demo
└── monitoring/ # Experiment tracking

## 👤 Author
Your Name - [LinkedIn](your-linkedin-url)