from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

class HfLlmLogicBrain:
    def __init__(self, hf_token):
    
      # Load the Llama 2 model and tokenizer
      # model_name = "meta-llama/Llama-2-7b-chat-hf"  # Example model name, adjust based on the specific Llama model
      # model_name = "gpt2"
      # model_name = "EleutherAI/gpt-neo-1.3B"
      # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      # self.model = AutoModelForCausalLM.from_pretrained(model_name)
      repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
      self.llm = HuggingFaceEndpoint(
          repo_id=repo_id,
          temperature=0.5,
          huggingfacehub_api_token=hf_token,
      )

    def respond(self, question):
      template = """"
      Answer the following question in a few sentences:
      
      {question}
      """

      prompt = PromptTemplate.from_template(template)

      
      llm_chain = prompt | self.llm
      response = llm_chain.invoke({"question": question})
      
      return response
      