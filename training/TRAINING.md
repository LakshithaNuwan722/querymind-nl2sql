# Training Details

## Model
- **Base**: unsloth/Llama-3.2-3B-Instruct
- **Method**: QLoRA (4-bit quantization)
- **Framework**: Unsloth + TRL

## Dataset
- **Source**: Spider NL2SQL benchmark
- **Train**: 3,000 samples
- **Val**: 300 samples

## Results
| Metric | Value |
|--------|-------|
| Training Loss | 0.2640 |
| Eval Loss | 0.7673 |
| Total Steps | 564 |
| Epochs | 3 |

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Batch size | 2 |
| Grad accumulation | 8 |
| Effective batch | 16 |
| Warmup steps | 100 |
| LR scheduler | cosine |

## Links
- Model : https://huggingface.co/lakshitha722/querymind-nl2sql
- W&B   : https://wandb.ai/lakshithanuwan722-other/querymind-nl2sql

## Training Environment
- Platform : Google Colab
- GPU : Tesla T4 (15GB)
- Training time : ~2-3 hours