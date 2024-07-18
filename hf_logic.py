import os
from huggingface_hub import HfApi
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

# Set the cache directory to /tmp
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['HF_HOME'] = '/tmp'

class HfLlmLogicBrain:
    def __init__(self, hf_token, pinecone_api_key, pinecone_index_name):
      api = HfApi(token=hf_token)
      user_info = api.whoami()
      print(f"Successfully authenticated as: {user_info['name']}")
    
      repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
      self.llm = HuggingFaceEndpoint(
          repo_id=repo_id,
          temperature=0.5,
          # huggingfacehub_api_token=hf_token,
      )
      
      print("Loaded LLM")
      
      embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
      
      print("Loaded embeddings")
      
      vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
      
      print("Loaded vector store")
      
      self.retriever = vectorstore.as_retriever()

    @staticmethod
    def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def clean_response(response):
      if "Question:" in response:
          response = response.split("Question:")[0].strip()
      return response

    def respond_with_chain(self, query):     
      template = """You are an assistant specialized in answering general questions about cats. 
      If asked about a cat available for adoption, use the following pieces of retrieved context to provide accurate and concise answers. 
      If asked a general question about cats or pets in general, do not use the retrieved context.
      Politely and cutely decline to answer questions about unrelated topics, such as politics. 
      Keep your answers to a maximum of three sentences and ensure they are concise.

      Context: {context}

      Question: {question}

      Helpful Answer:"""
      custom_rag_prompt = PromptTemplate.from_template(template)

      rag_chain = (
          {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
          | custom_rag_prompt
          | self.llm
          | StrOutputParser()
      )

      return self.clean_response(rag_chain.invoke(query))
      
      

    def respond(self, question):
      template = """"
      Answer the following question in a few sentences:
      
      {question}
      """

      prompt = PromptTemplate.from_template(template)

      
      llm_chain = prompt | self.llm
      response = llm_chain.invoke({"question": question})
      
      return response
      