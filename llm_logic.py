from langchain_huggingface import HuggingFaceEndpoint

class LlmLogicBrain:
    def __init__(self, pinecone_index_name):
      repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
      self.llm = HuggingFaceEndpoint(
          repo_id=repo_id,
          temperature=0.2,
      )
      
      print("Loaded LLM")
      
      return None

    def respond_with_chain(self, query):
      return self.llm.invoke(query)
