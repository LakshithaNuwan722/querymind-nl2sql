"""  
Day 10 - Gradio Demo  
Interactive NL to SQL demo  
"""  
  
import os  
import torch  
from dotenv import load_dotenv  
  
load_dotenv()  

# 🚀 Force Unsloth to use standard, stable HuggingFace download engine (essential for VPN/WARP and Windows proxies)
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
  
print("=" * 60)  
print("LOADING QUERYMIND DEMO")  
print("=" * 60)  
  
  
# ─────────────────────────────────────────  
# 1. LOAD MODEL  
# ─────────────────────────────────────────  
  
MODEL_NAME = "lakshitha722/querymind-nl2sql"  
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"  
  
print(f"\nLoading model: {MODEL_NAME}")  
print(f"Device: {DEVICE}")  
  
from unsloth import FastLanguageModel  
  
model, tokenizer = FastLanguageModel.from_pretrained(  
    model_name     = MODEL_NAME,  
    max_seq_length = 1024,  
    load_in_4bit   = True,  
    dtype          = None,  
)  
FastLanguageModel.for_inference(model)  
print("✅ Model loaded successfully!")  
  
  
# ─────────────────────────────────────────  
# 2. INFERENCE FUNCTION  
# ─────────────────────────────────────────  
  
PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.  
  
### Instruction:  
Convert the following natural language question to a SQL query based on the given database schema. Return ONLY the SQL query, nothing else.  
  
### Schema:  
{schema}  
  
### Question:  
{question}  
  
### Response:  
"""  
  
def generate_sql(question: str, schema: str) -> tuple:  
    """Generate SQL from question and schema"""  
    import time  
    import re  
  
    if not question.strip():  
        return "Please enter a question", "0 ms"  
  
    if not schema.strip():  
        schema = "Database: unknown"  
  
    prompt = PROMPT_TEMPLATE.format(  
        schema   = schema,  
        question = question,  
    )  
  
    inputs = tokenizer(  
        [prompt],  
        return_tensors = "pt",  
    ).to(DEVICE)  
  
    start = time.time()  
  
    with torch.no_grad():  
        outputs = model.generate(  
            **inputs,  
            max_new_tokens = 150,  
            temperature    = 0.1,  
            do_sample      = False,  
            pad_token_id   = tokenizer.eos_token_id,  
        )  
  
    latency_ms   = (time.time() - start) * 1000  
    input_length = inputs['input_ids'].shape[1]  
  
    generated = tokenizer.decode(  
        outputs[0][input_length:],  
        skip_special_tokens = True,  
    ).strip()  
  
    # Clean output  
    generated = re.sub(r'```sql\s*', '', generated, flags=re.IGNORECASE)  
    generated = re.sub(r'```\s*',    '', generated)  
    sql       = generated.split('\n')[0].strip()  
    sql       = sql.rstrip(';').strip()  
  
    latency_str = f"{latency_ms:.0f} ms"  
  
    return sql, latency_str  
  
  
# ─────────────────────────────────────────  
# 3. EXAMPLE QUERIES  
# ─────────────────────────────────────────  
  
EXAMPLES = [  
    [  
        "How many employees are there?",  
        "Database: company\nTables: employees (id, name, department, salary, hire_date)",  
    ],  
    [  
        "What is the average salary by department?",  
        "Database: hr\nTables: employees (id, name, department, salary)",  
    ],  
    [  
        "List all customers who placed more than 5 orders",  
        "Database: sales\nTables: customers (id, name, email), orders (id, customer_id, amount, date)",  
    ],  
    [  
        "Find the top 3 products by total revenue",  
        "Database: store\nTables: products (id, name, price), order_items (id, product_id, quantity)",  
    ],  
    [  
        "How many orders were placed each month this year?",  
        "Database: sales\nTables: orders (id, customer_id, amount, order_date)",  
    ],  
    [  
        "What is the maximum salary in each department?",  
        "Database: hr\nTables: employees (id, name, department, salary, manager_id)",  
    ],  
]  
  
  
# ─────────────────────────────────────────  
# 4. BUILD GRADIO UI  
# ─────────────────────────────────────────  
  
print("\nBuilding Gradio UI...")  
  
# Custom CSS  
CSS = """  
.container { max-width: 900px; margin: auto; }  
.title { text-align: center; }  
.sql-output { font-family: monospace; font-size: 14px; }  
"""  
  
# We'll need gradio installed to run this script.
try:
    import gradio as gr  
except ImportError:
    print("❌ Error: 'gradio' package not found. Install it with: pip install gradio")
    exit(1)

with gr.Blocks(  
    css   = CSS,  
    title = "QueryMind - NL to SQL"  
) as demo:  
  
    # Header  
    gr.Markdown("""  
    # 🧠 QueryMind: Natural Language to SQL  
  
    > Fine-tuned LLaMA 3.2 3B that converts plain English to SQL queries  
  
    **Model**: [lakshitha722/querymind-nl2sql](https://huggingface.co/lakshitha722/querymind-nl2sql)  
    """)  
  
    with gr.Row():  
  
        # Left column - inputs  
        with gr.Column(scale=1):  
  
            question_input = gr.Textbox(  
                label       = "📝 Your Question",  
                placeholder = "e.g. How many employees are there?",  
                lines       = 3,  
            )  
  
            schema_input = gr.Textbox(  
                label       = "🗄️ Database Schema",  
                placeholder = "e.g. Database: company\nTables: employees (id, name, salary)",  
                lines       = 6,  
                value       = "Database: company\nTables: employees (id, name, department, salary, hire_date)",  
            )  
  
            with gr.Row():  
                generate_btn = gr.Button(  
                    "⚡ Generate SQL",  
                    variant = "primary",  
                    size    = "lg",  
                )  
                clear_btn = gr.Button(  
                    "🗑️ Clear",  
                    variant = "secondary",  
                    size    = "lg",  
                )  
  
        # Right column - outputs  
        with gr.Column(scale=1):  
  
            sql_output = gr.Code(  
                label    = "📊 Generated SQL",  
                language = "sql",  
                lines    = 8,  
            )  
  
            latency_output = gr.Textbox(  
                label     = "⏱️ Generation Time",  
                lines     = 1,  
                interactive = False,  
            )  
  
            gr.Markdown("""  
            ### 💡 Tips  
            - Provide clear schema with table and column names  
            - Ask specific questions for better results  
            - Model works best with Spider-style questions  
            """)  
  
    # Examples  
    gr.Markdown("## 📚 Example Queries")  
    gr.Examples(  
        examples         = EXAMPLES,  
        inputs           = [question_input, schema_input],  
        outputs          = [sql_output, latency_output],  
        fn               = generate_sql,  
        cache_examples   = False,  
        label            = "Click any example to try it",  
    )  
  
    # Footer  
    gr.Markdown("""  
    ---  
    **Training Details**: Fine-tuned on Spider NL2SQL dataset |  
    **Method**: QLoRA (4-bit) |  
    **Loss**: 0.2640 |  
    **Samples**: 3,000  
    """)  
  
    # Button actions  
    generate_btn.click(  
        fn      = generate_sql,  
        inputs  = [question_input, schema_input],  
        outputs = [sql_output, latency_output],  
    )  
  
    clear_btn.click(  
        fn      = lambda: ("", "", ""),  
        inputs  = [],  
        outputs = [question_input, sql_output, latency_output],  
    )  
  
print("✅ Gradio UI ready!")  
  
  
# ─────────────────────────────────────────  
# 5. LAUNCH  
# ─────────────────────────────────────────  
  
if __name__ == "__main__":  
    demo.launch(  
        share          = True,   # Creates public URL  
        server_name    = "0.0.0.0",  
        server_port    = 7860,  
        show_error     = True,  
    )
