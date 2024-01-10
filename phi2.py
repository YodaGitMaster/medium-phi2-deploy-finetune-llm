# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

# Set environment variables for Hugging Face models
hf_home_path: str = os.getcwd()  # Set HF_HOME to the current working directory
hf_hub_home_path: str = os.path.join(os.getcwd(), ".cache")  # Set HF_HUB_HOME to a subdirectory in the current working directory
disable_symlinks_warning: str = "1"  # Disable the warning for symbolic links

os.environ["HF_HOME"] = hf_home_path
os.environ["HF_HUB_HOME"] = hf_hub_home_path
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = disable_symlinks_warning

# Define a function to load the model and tokenizer
def load_model_and_tokenizer() -> tuple:
    '''
    This function loads the model and tokenizer.
    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    '''
    # Specify the model name
    model_name: str = 'microsoft/phi-2'
    # Load the model with specific configurations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",  # Automatically determines the data type
        flash_attn=True,  # Enables the flash attention mechanism
        # flash_rotary=True,  # Enables the flash rotary mechanism
        # fused_dense=True,  # Enables the fused dense mechanism
        device_map="cuda",  # Specifies the device map to be used
        trust_remote_code=True  # Trusts the remote code
    
    )
    # Load the tokenizer corresponding to the model
    tokenizer = AutoTokenizer.from_pretrained(model_name,  use_fast = True)

    # Set the padding token to the EOS token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Sets the padding token to the end-of-sentence token

    # Return the loaded model and tokenizer
    return model, tokenizer

# Define a function to generate text based on a given prompt
def generate_text(prompt: str, model, tokenizer, max_length: int = 180, temperature: float = 0.5,
                  num_return_sequences: int = 1, top_p: float = 0.95, top_k: int = 20,
                  repetition_penalty: float = 1.1) -> str:
    '''
    This function generates text based on a given prompt.
    Returns:
        str: The generated text.
    '''
    # Encode the prompt using the tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors='pt')  # Encodes the prompt into input IDs

    # Ensure the model and input are on the same device
    input_ids = input_ids.to(model.device)  # Moves the input IDs to the same device as the model

    # Generate a response using the model
    with torch.no_grad():  # Disables gradient calculation
        output = model.generate(  # Generates a response
            input_ids,  # The input IDs
            max_length=max_length,  # The maximum length of the generated text
            temperature=temperature,  # The temperature for the generation process
            do_sample=True,  # Enables sampling
            num_return_sequences=num_return_sequences,  # The number of sequences to return
            top_p=top_p,  # The top-p parameter for nucleus sampling
            top_k=top_k,  # The top-k parameter for top-k sampling
            repetition_penalty=repetition_penalty  # The repetition penalty
        )

    # Decode the model output to get the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)  # Decodes the output to get the generated text

    # Return the generated text
    return generated_text

import re

def split_text_at_pattern(text):
    # Define the pattern for lines starting with a word in all uppercase letters followed by a colon
    pattern = r'^[A-Z][a-zA-Z\s]*:'

    # Find all matches of the pattern in the text
    matches = re.finditer(pattern, str(text), re.MULTILINE)
    print(matches)
    
    print('-'*100)
    print(text)
    print('-'*100)

    # If there is at least one match, split the text at the first match
    if matches:
        match = next(matches)
        split_result = re.split(re.escape(match.group()), text, 1, flags=re.MULTILINE)
        
        # Return the first half of the split result
        if len(split_result) > 1:
            return split_result[0].strip()  # Strip extra whitespace

    # Return None if the pattern doesn't match or the split result is not available
    return None

# Define the main function
def main():
    '''
    This is the main function.
    '''
    # Load the pre-trained model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Define the user instruction as a prompt
    user_instruction: str = "Write a 'Hello, World!' function in Rust and give instructions on how to run the program"

    # Construct the prompt using user instruction
    prompt = f'''Instruct:{user_instruction}\nOutput:'''
    # Generate and print the response with specific parameter values
    result: str = generate_text(
        prompt,
        model,
        tokenizer,
        max_length=200,  # Assign a specific value for max_length
        temperature=0.8,  # Assign a specific value for temperature
        num_return_sequences=2,  # Assign a specific value for num_return_sequences
        top_p=0.8,  # Assign a specific value for top_p
        top_k=30,  # Assign a specific value for top_k
        repetition_penalty=1.2  # Assign a specific value for repetition_penalty
    )

    split_text = result.split("Output:",1)
    assistant_response = split_text[1].strip() if len(split_text) > 1 else ""
    assistant_response = assistant_response.replace("<|endoftext|>", "").strip()
    try:
        assistant_response = split_text_at_pattern(assistant_response)
        print(f">>> {assistant_response}")
        return assistant_response
    except:
        print(f">>> {assistant_response}")
        return assistant_response

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
