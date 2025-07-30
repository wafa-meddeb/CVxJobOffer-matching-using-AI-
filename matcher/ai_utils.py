from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import json, re
from functools import lru_cache

# Load Zephyr model
@lru_cache()
def get_llm_pipeline():
    model_id = 'HuggingFaceH4/zephyr-7b-alpha'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
    return pipeline('text-generation', model=model, tokenizer=tokenizer)

def build_prompt_job_offer(text):
    return f"""You are a helpful HR assistant. Extract this job offer into JSON:
{text}
Return:
- title, company, location, description
- requirements (list)
- nice_to_have (list)
- contract_type, experience_level
- languages (list), technologies (list), salary_range

JSON:
"""

def extract_json_from_text(text: str) -> dict:
    start = text.find('{')
    if start == -1:
        raise ValueError("No opening brace found.")

    brace_count = 0
    for i, char in enumerate(text[start:], start=start):
        if char == '{': brace_count += 1
        elif char == '}': brace_count -= 1
        if brace_count == 0:
            json_str = text[start:i + 1]
            break
    else:
        raise ValueError("Unbalanced braces")

    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Error parsing JSON: {e}")

def parse_with_llm(prompt):
    pipe = get_llm_pipeline()
    output = pipe(prompt, max_new_tokens=800, temperature=0.3)[0]['generated_text']
    return extract_json_from_text(output.strip())

