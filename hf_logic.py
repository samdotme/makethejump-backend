from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

class HfLlmLogicBrain:
    def __init__(self, hf_token):
      
      print(f"Token obtained from environment: {hf_token}")
    
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
      