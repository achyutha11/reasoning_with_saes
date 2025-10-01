from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer
