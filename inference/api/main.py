"""  
Day 11 - FastAPI Inference API  
Production-ready SQL generation endpoint  
"""  
  
import os  
import time  
import torch  
from fastapi import FastAPI, HTTPException  
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel  
from dotenv import load_dotenv  
from contextlib import asynccontextmanager  
  
load_dotenv()  

# 🚀 Force Unsloth to use standard, stable HuggingFace download engine (essential for VPN/WARP and Windows proxies)
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
  
  
# ─────────────────────────────────────────  
# 1. MODEL GLOBALS  
# ─────────────────────────────────────────  
  
model     = None  
tokenizer = None  
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"  
  
MODEL_NAME = "lakshitha722/querymind-nl2sql"  
  
PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.  
  
### Instruction:  
Convert the following natural language question to a SQL query based on the given database schema. Return ONLY the SQL query, nothing else.  
  
### Schema:  
{schema}  
  
### Question:  
{question}  
  
### Response:  
"""  
  
  
# ─────────────────────────────────────────  
# 2. STARTUP + SHUTDOWN  
# ─────────────────────────────────────────  
  
@asynccontextmanager  
async def lifespan(app: FastAPI):  
    """Load model on startup"""  
    global model, tokenizer  
  
    print("Loading model on startup...")  
  
    from unsloth import FastLanguageModel  
  
    model, tokenizer = FastLanguageModel.from_pretrained(  
        model_name     = MODEL_NAME,  
        max_seq_length = 1024,  
        load_in_4bit   = True,  
        dtype          = None,  
    )  
    FastLanguageModel.for_inference(model)  
  
    print(f"✅ Model loaded: {MODEL_NAME}")  
    print(f"✅ Device: {DEVICE}")  
  
    yield  
  
    print("Shutting down...")  
  
  
# ─────────────────────────────────────────  
# 3. FASTAPI APP  
# ─────────────────────────────────────────  
  
app = FastAPI(  
    title       = "QueryMind API",  
    description = "Natural Language to SQL using fine-tuned LLaMA 3.2",  
    version     = "1.0.0",  
    lifespan    = lifespan,  
)  
  
# CORS  
app.add_middleware(  
    CORSMiddleware,  
    allow_origins  = ["*"],  
    allow_methods  = ["*"],  
    allow_headers  = ["*"],  
)  
  
  
# ─────────────────────────────────────────  
# 4. REQUEST/RESPONSE MODELS  
# ─────────────────────────────────────────  
  
class SQLRequest(BaseModel):  
    question   : str  
    schema     : str  
    max_tokens : int = 150  
  
    class Config:  
        json_schema_extra = {  
            "example": {  
                "question"   : "How many employees are there?",  
                "schema"     : "Database: company\nTables: employees (id, name, salary)",  
                "max_tokens" : 150,  
            }  
        }  
  
  
class SQLResponse(BaseModel):  
    sql         : str  
    valid       : bool  
    latency_ms  : float  
    tokens_used : int  
    model       : str  
  
  
class HealthResponse(BaseModel):  
    status  : str  
    model   : str  
    device  : str  
    version : str  
  
  
# ─────────────────────────────────────────  
# 5. HELPER FUNCTIONS  
# ─────────────────────────────────────────  
  
def clean_sql_output(raw: str) -> str:  
    """Clean model output"""  
    import re  
  
    if not raw:  
        return ""  
  
    cleaned = re.sub(r'```sql\s*', '', raw, flags=re.IGNORECASE)  
    cleaned = re.sub(r'```\s*',    '', cleaned)  
  
    lines = cleaned.strip().split('\n')  
    sql_lines = []  
  
    for line in lines:  
        line = line.strip()  
        if not line:  
            continue  
        if sql_lines and re.match(  
            r'^(This|The|Note|Here)', line, re.IGNORECASE  
        ):  
            break  
        sql_lines.append(line)  
  
    result = ' '.join(sql_lines).strip()  
    result = result.rstrip(';').strip()  
  
    return result  
  
  
def is_valid_sql(sql: str) -> bool:  
    """Check SQL validity"""  
    if not sql or len(sql.strip()) < 5:  
        return False  
    sql_upper = sql.upper()  
    return "SELECT" in sql_upper and "FROM" in sql_upper  
  
  
# ─────────────────────────────────────────  
# 6. API ENDPOINTS  
# ─────────────────────────────────────────  
  
@app.get("/", tags=["General"])  
async def root():  
    """API root"""  
    return {  
        "name"        : "QueryMind API",  
        "version"     : "1.0.0",  
        "description" : "Natural Language to SQL",  
        "endpoints"   : {  
            "health"  : "/health",  
            "predict" : "/generate-sql",  
            "docs"    : "/docs",  
        }  
    }  
  
  
@app.get(  
    "/health",  
    response_model = HealthResponse,  
    tags           = ["General"]  
)  
async def health():  
    """Health check"""  
    return HealthResponse(  
        status  = "healthy",  
        model   = MODEL_NAME,  
        device  = DEVICE,  
        version = "1.0.0",  
    )  
  
  
@app.post(  
    "/generate-sql",  
    response_model = SQLResponse,  
    tags           = ["SQL Generation"]  
)  
async def generate_sql(request: SQLRequest):  
    """  
    Generate SQL from natural language question  
  
    - **question**: Natural language question  
    - **schema**: Database schema description  
    - **max_tokens**: Maximum tokens to generate  
    """  
    global model, tokenizer  
  
    if model is None:  
        raise HTTPException(  
            status_code = 503,  
            detail      = "Model not loaded"  
        )  
  
    if not request.question.strip():  
        raise HTTPException(  
            status_code = 400,  
            detail      = "Question cannot be empty"  
        )  
  
    # Build prompt  
    prompt = PROMPT_TEMPLATE.format(  
        schema   = request.schema or "Database: unknown",  
        question = request.question,  
    )  
  
    # Tokenize  
    inputs = tokenizer(  
        [prompt],  
        return_tensors = "pt",  
    ).to(DEVICE)  
  
    # Generate  
    start = time.time()  
  
    with torch.no_grad():  
        outputs = model.generate(  
            **inputs,  
            max_new_tokens = request.max_tokens,  
            temperature    = 0.1,  
            do_sample      = False,  
            pad_token_id   = tokenizer.eos_token_id,  
        )  
  
    latency_ms   = (time.time() - start) * 1000  
    input_length = inputs['input_ids'].shape[1]  
    tokens_used  = outputs.shape[1] - input_length  
  
    # Decode  
    raw_output = tokenizer.decode(  
        outputs[0][input_length:],  
        skip_special_tokens = True,  
    ).strip()  
  
    # Clean  
    sql   = clean_sql_output(raw_output)  
    valid = is_valid_sql(sql)  
  
    return SQLResponse(  
        sql         = sql,  
        valid       = valid,  
        latency_ms  = round(latency_ms, 1),  
        tokens_used = tokens_used,  
        model       = MODEL_NAME,  
    )  
  
  
@app.get(  
    "/examples",  
    tags = ["SQL Generation"]  
)  
async def get_examples():  
    """Get example queries"""  
    return {  
        "examples": [  
            {  
                "question" : "How many employees are there?",  
                "schema"  : "Database: company\nTables: employees (id, name, dept, salary)",  
            },  
            {  
                "question" : "What is the average salary by department?",  
                "schema"   : "Database: hr\nTables: employees (id, name, department, salary)",  
            },  
            {  
                "question" : "List top 5 customers by total orders",  
                "schema"   : "Database: sales\nTables: customers (id, name), orders (id, customer_id, amount)",  
            },  
        ]  
    }  
