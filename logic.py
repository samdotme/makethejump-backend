from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

class LlmLogicBrain:
    def __init__(self, hf_token):
    
      # Load the Llama 2 model and tokenizer
      # model_name = "meta-llama/Llama-2-7b-chat-hf"  # Example model name, adjust based on the specific Llama model
      # model_name = "gpt2"
      model_name = "EleutherAI/gpt-neo-1.3B"
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def respond(self, prompt):
      context_prompt = f"Give a suggestion for the following prompt: \n\n\"{prompt}\"\n\nSuggestion:"
     
     
      # Tokenize the input
      inputs = self.tokenizer(prompt, return_tensors="pt")
      
      print(inputs)

      # Generate response
      with torch.no_grad():
          outputs = self.model.generate(**inputs,
              max_new_tokens=50,
              temperature=0.7,
              top_k=50,
              top_p=0.95,
              repetition_penalty=2.5,  # Increased repetition penalty
              do_sample=True,  # Enable sampling for more creative responses
              num_return_sequences=1  # Return a single response
          )
                        
      print(outputs)

      # Decode the generated tokens
      response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
      
      return response
      