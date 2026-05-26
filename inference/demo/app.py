"""  
HuggingFace Spaces deployment file  
Must be named app.py for HF Spaces  
"""  
  
import os  
import re  
import torch  
import gradio as gr  
  
print("Starting QueryMind Demo...")  
  
# ─────────────────────────────────────────  
# LOAD MODEL  
# ─────────────────────────────────────────  
  
MODEL_NAME = "lakshitha722/querymind-nl2sql"  
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"  
  
print(f"Loading {MODEL_NAME} on {DEVICE}...")  
  
try:  
    # 🚀 If running on a GPU Space, Unsloth will be blazing fast!
    from unsloth import FastLanguageModel  
  
    model, tokenizer = FastLanguageModel.from_pretrained(  
        model_name     = MODEL_NAME,  
        max_seq_length = 1024,  
        load_in_4bit   = True if DEVICE == "cuda" else False,  
        dtype          = None,  
    )  
    FastLanguageModel.for_inference(model)  
    print("✅ Loaded successfully with Unsloth!")  
  
except Exception as e:  
    print(f"⚠️ Unsloth not available or failed: {e}")  
    print("Falling back to standard HuggingFace transformers...")  
    
    from transformers import AutoModelForCausalLM, AutoTokenizer  
    from peft import PeftModel  
  
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)  
    
    # 💡 Use 'unsloth/Llama-3.2-3B-Instruct' instead of 'meta-llama' 
    # to avoid Gated Model Token requirement errors on HF Spaces!
    base_model_name = "unsloth/Llama-3.2-3B-Instruct"
    
    if DEVICE == "cuda":
        # On GPU Space, load base model in 16-bit
        base_model = AutoModelForCausalLM.from_pretrained(  
            base_model_name,  
            torch_dtype = torch.float16,  
            device_map  = "auto",  
        )  
    else:
        # On free CPU Space (16GB RAM limit), load in 8-bit or bfloat16 
        # to prevent crashing (OOM - Out of Memory / Exit Code 137)
        print("Running on CPU Space. Loading in bfloat16 to optimize memory usage...")
        base_model = AutoModelForCausalLM.from_pretrained(  
            base_model_name,  
            torch_dtype = torch.bfloat16,  
            device_map  = "auto",  
        )  
        
    model = PeftModel.from_pretrained(base_model, MODEL_NAME)  
    model.eval()  
    print("✅ Loaded successfully with transformers fallback!")  
  
  
# ─────────────────────────────────────────  
# INFERENCE  
# ─────────────────────────────────────────  
  
PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.  
  
### Instruction:  
Convert the following natural language question to a SQL query based on the given database schema. Return ONLY the SQL query, nothing else.  
  
### Schema:  
{schema}  
  
### Question:  
{question}  
  
### Response:  
"""  
  
def predict(question: str, schema: str) -> tuple:  
    """Generate SQL prediction"""  
    import time  
  
    if not question.strip():  
        return "Please enter a question", "0 ms"  
  
    prompt = PROMPT.format(  
        schema   = schema or "Database: unknown",  
        question = question,  
    )  
  
    inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)  
  
    start = time.time()  
    with torch.no_grad():  
        outputs = model.generate(  
            **inputs,  
            max_new_tokens = 150,  
            temperature    = 0.1,  
            do_sample      = False,  
            pad_token_id   = tokenizer.eos_token_id,  
        )  
    latency = (time.time() - start) * 1000  
  
    input_len = inputs['input_ids'].shape[1]  
    generated = tokenizer.decode(  
        outputs[0][input_len:],  
        skip_special_tokens=True,  
    ).strip()  
  
    # Clean  
    generated = re.sub(r'```sql\s*', '', generated, flags=re.IGNORECASE)  
    generated = re.sub(r'```\s*',    '', generated)  
    sql       = generated.split('\n')[0].strip().rstrip(';')  
  
    return sql, f"{latency:.0f} ms"  
  
  
# ─────────────────────────────────────────  
# GRADIO UI  
# ─────────────────────────────────────────  
  
EXAMPLES = [  
    ["How many employees are there?",  
     "Database: company\nTables: employees (id, name, department, salary)"],  
    ["What is the average salary by department?",  
     "Database: hr\nTables: employees (id, name, department, salary)"],  
    ["List top 5 customers by order count",  
     "Database: sales\nTables: customers (id, name), orders (id, customer_id, date)"],  
    ["Find products with price greater than 100",  
     "Database: store\nTables: products (id, name, price, category)"],  
]  
  
with gr.Blocks(title="QueryMind - NL to SQL") as demo:  
  
    gr.Markdown("""  
    # 🧠 QueryMind: Natural Language → SQL  
    Fine-tuned LLaMA 3.2 3B | Training Loss: 0.2640 | Dataset: Spider  
    """)  
  
    with gr.Row():  
        with gr.Column():  
            question = gr.Textbox(  
                label       = "Your Question",  
                placeholder = "How many employees are there?",  
                lines       = 2,  
            )  
            schema = gr.Textbox(  
                label       = "Database Schema",  
                placeholder = "Database: company\nTables: employees (id, name, salary)",  
                lines       = 4,  
                value       = "Database: company\nTables: employees (id, name, department, salary)",  
            )  
            btn = gr.Button("Generate SQL ⚡", variant="primary")  
  
        with gr.Column():  
            sql_out     = gr.Code(label="Generated SQL", language="sql")  
            latency_out = gr.Textbox(label="Latency")  
  
    gr.Examples(  
        examples = EXAMPLES,  
        inputs   = [question, schema],  
        outputs  = [sql_out, latency_out],  
        fn       = predict,  
    )  
  
    btn.click(  
        fn      = predict,  
        inputs  = [question, schema],  
        outputs = [sql_out, latency_out],  
    )  
  
demo.launch()
