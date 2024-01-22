from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import warnings
import re
import time

warnings.filterwarnings("ignore")

hf_home_path: str = os.getcwd()
hf_hub_home_path: str = os.path.join(os.getcwd(), ".cache")
disable_symlinks_warning: str = "1"
os.environ["HF_HOME"] = hf_home_path
os.environ["HF_HUB_HOME"] = hf_hub_home_path
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = disable_symlinks_warning


def load_model_and_tokenizer() -> tuple:
    model_name: str = 'microsoft/phi-2'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(f"-------\nModel Name {model_name}\nVocabolary size in use: {tokenizer.vocab_size}\n------")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_text(prompt: str, model, tokenizer, top_p: float = 0.95, top_k: int = 20, max_length: int = 500, temperature: float = 0.5,
                  num_return_sequences: int = 1) -> str:
    
    # input_ids = tokenizer.encode(prompt, return_tensors='pt', padding=True, truncation=True)
    # input_ids = input_ids.to(model.device)
    
    tokens = tokenizer.encode(prompt, padding=True, truncation=True, return_tensors='pt').to('cuda')
    
    length = len(tokens[0])
    with torch.no_grad():
        output = model.generate(
            tokens,
            max_length=length + max_length,
            use_cache=True,
            do_sample=False,
            top_p=top_p,
            temperature=temperature,
            top_k=top_k,
        )
            

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def find_all_occurrences(pattern, text):
    matches = re.finditer(pattern, text)
    occurrences = []

    for match in matches:
        occurrence_start = match.start()
        occurrence_end = match.end()
        occurrences.append(str(text[occurrence_start:occurrence_end]).replace('\n', ' '))

    return occurrences


def main(user_prompt: str,schema:str, difficulty: int):
    model, tokenizer = load_model_and_tokenizer()
    
    system_prompt = '''You are an AI assistant that follows instruction extremely well.
                            '''
                        
    prompt = f'''Instruct:{system_prompt}</s>User:{user_prompt}\n</s>Schema:{schema}</s>\nOutput:'''
    
    result = None
    ripasso = 0
    max_turns = 0
    sql_queries = set()
    sql_query = ''

    while ripasso <= difficulty and max_turns <= 5:
        temperature = -0.1 + (((ripasso + 1) / 10))
        top_p = 1.20 - (((ripasso + 1) / 10))
        top_k = ripasso + 1 * 10 
        
        if top_p <=0:
            top_p = 0.1
        if temperature <=0:
            temperature = 0.1
            
        result = generate_text(
            prompt,
            model,
            tokenizer,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_length=200,
            num_return_sequences=1
        )
        
    return result


