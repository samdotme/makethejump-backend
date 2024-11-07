from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

class LlmLogicBrain:
    def __init__(self, pinecone_index_name):
      repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
      self.llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.9
      )
      
      print("Loaded LLM")
      
      return None

    def respond_with_chain(self, query): 
      context = """You are a virtual assistant specialized in answering questions about cats. 
      If a question is unrelated to cats, pets in general, or pet adoption, politely and cutely decline to answer, 
      explaining that you only provide information about these topics. 
      Keep your responses to a maximum of three sentences, and ensure they are concise and friendly"""
      
      template = """
      Context: {context}

      Question: {question}

      Helpful Answer:"""

      prompt = PromptTemplate.from_template(template)
      
      chain = prompt | self.llm

      # Example invocation:
      return chain.invoke({"context": context, "question": query})
