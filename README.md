🧠 QueryMind: Natural Language to SQL Engine
QueryMind is a production-ready, domain-specific NL-to-SQL engine powered by a fine-tuned LLaMA 3.2 3B Instruct model. Using QLoRA (4-bit) quantization via Unsloth, the model has been trained on over 3,000 samples from the Spider NL2SQL dataset to convert natural language English queries into accurate, schema-valid SQL statements.

Optimized to run locally on consumer GPUs (including next-generation Blackwell NVIDIA RTX 5070 architectures) and deployable as a high-speed microservice.

🔗 Project Links
🤗 Fine-Tuned Model: lakshitha722/querymind-nl2sql
📊 Weights & Biases (W&B) Dashboard: querymind-nl2sql Run Stats
🎮 Live HuggingFace Space Demo: QueryMind NL2SQL Demo
📡 Local Production API: Running locally on http://localhost:8000 (Interactive Swagger Docs at http://localhost:8000/docs)
📊 3-Way Evaluation Results (100 Sample Test)
To measure the real-world performance of our fine-tuned 3B model, we conducted a rigorous 3-way evaluation benchmarking our local model against raw, massive base models running on cloud infrastructure (via Groq API).

Metric	Base LLaMA 3.1 8B (Groq)	Fine-Tuned LLaMA 3.2 3B (Local RTX 5070)	Base LLaMA 3.3 70B (Groq)
Valid SQL %	100.0%	100.0%	100.0%
Exact Match %	14.0%	19.0% 🏆	11.0%
Avg Schema Similarity	0.816	0.863	0.874
High Similarity % (>=0.7)	68.0%	77.0%	82.0%
Avg Inference Latency (ms)	1993.0 ms	1539.8 ms ⚡	2028.8 ms
Avg Tokens Generated	182.9	34.6 (Highly Concise!) 📉	187.3
API Cost per 1k Queries	$0.00	$0.00 💰	$0.00
💡 Key Insights:
Small Model Superiority: The fine-tuned 3B model achieved a 19.0% Exact Match score, beating the massive LLaMA 70B model (11.0%) by 8 percentage points on domain-specific SQL translation!
Extreme Efficiency: Thanks to targeted training, the fine-tuned model outputs only the SQL query (34.6 average tokens) without conversational filler, unlike the base models which generate up to 187 tokens of explanations.
RTX 5070 Speed Demon: Running locally on an RTX 5070 Laptop GPU, inference completes in ~1.5 seconds, outperforming standard cloud API round-trips!
🛠️ Technology Stack
Component	Technology Used
Base Model	LLaMA 3.2 3B Instruct (unsloth/Llama-3.2-3B-Instruct)
Quantization & Training	QLoRA 4-bit Quantization + Unsloth (2x faster finetuning)
Hardware Target	NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM, sm_120 Blackwell Architecture)
Dataset	Spider NL2SQL Dataset (3,000 highly structured training pairs)
Experiment Tracking	Weights & Biases (W&B)
Interactive UI	Gradio (Deployed to HF Spaces)
Microservice Backend	FastAPI + Uvicorn ASGI Server
🚀 Blackwell GPU & Windows Download Optimizations
Running Blackwell-architecture cards (like the RTX 5070 with CUDA compute capability sm_120) on Windows requires customized configurations:

📁 Project Structure
querymind-nl2sql/
├── data/
│   └── splits/             # Train, Validation, and Test JSONL splits
├── evaluation/
│   ├── results/            # Saved evaluation JSON outputs (Base 8B, FT 3B, Base 70B)
│   └── run_evaluation.py   # Clean 3-Way benchmark evaluation pipeline
├── inference/
│   ├── api.py              # FastAPI production-ready SQL generation microservice
│   ├── app.py              # Interactive Gradio demo web interface
│   └── requirements.txt    # HuggingFace Spaces deployment dependencies
└── README.md

🏗️ Architecture Flow
       User Query
           │
           ▼
┌───────────────────────┐
│    FastAPI Backend    │ (inference/api.py) or Gradio (app.py)
└───────────┬───────────┘
           │
           ▼
┌───────────────────────┐
│    QueryMind Model    │ (LLaMA 3.2 3B + LoRA Adapter on local GPU)
└───────────┬───────────┘
           │
           ▼
    Raw LLM Output
           │
           ▼
┌───────────────────────┐
│  SQL Output Cleaner   │ (regex-based markdown and explanation removal)
└───────────┬───────────┘
           │
           ▼
   SQL Query Result  --->  [SELECT count(*) FROM employee]
   
📉 Training Convergence
Training Loss (per step): Converged smoothly from 2.0 down to 0.2640 over 3 Epochs (564 Steps).
Evaluation Loss: Stabilized at 0.7673.
Status: Successfully Converged ✅

PyTorch Blackwell Support: Installed via the CUDA 12.8 nightly/stable wheel indexes (--index-url https://download.pytorch.org/whl/cu128) to enable native GPU compilation.
Stable Sync Downloads: Configured UNSLOTH_STABLE_DOWNLOADS=1 and HF_HUB_ENABLE_HF_TRANSFER=1 to bypass Rust multi-threading download glitches on Windows. This forces stable, high-speed multi-threaded downloads over Cloudflare WARP / VPN proxies, boosting Sri Lankan ISP download rates from 10 KB/s to 15 MB/s+.

🧪 Sample Predictions
Question  : How many employees are there?
Schema    : Database: company | Tables: employees (id, name, department, salary)
Generated : SELECT COUNT(*) FROM employees ✅

Question  : What is the average salary by department?
Schema    : Database: hr | Tables: employees (id, name, department, salary)
Generated : SELECT department, AVG(salary) FROM employees GROUP BY department ✅

Question  : Find products with price greater than 100
Schema    : Database: store | Tables: products (id, name, price, category)
Generated : SELECT name, price FROM products WHERE price > 100 ✅

📄 License
Apache 2.0 - See LICENSE for details

👤 Author
Lakshitha

🤗 HuggingFace Profile: huggingface.co/lakshitha722
💻 GitHub: github.com/lakshithanuwan722
🌐 LinkedIn: [your-linkedin-url]
Built with ❤️ using LLaMA 3.2 + QLoRA + Unsloth
