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
        flash_rotary=True,  # Enables the flash rotary mechanism
        fused_dense=True,  # Enables the fused dense mechanism
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
            num_return_sequences=num_return_sequences  # The number of sequences to return
        )

    # Decode the model output to get the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)  # Decodes the output to get the generated text

    # Return the generated text
    return generated_text

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
    prompt = f'''### Human: {user_instruction}

### Assistant:
'''

    # Generate and print the response
    result: str = generate_text(prompt, model, tokenizer)  # Generates the response
    print(f">>> {result}")  # Prints the response
    return result

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
