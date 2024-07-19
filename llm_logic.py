from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

class LlmLogicBrain:
    def __init__(self, pinecone_index_name):
      repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
      self.llm = HuggingFaceEndpoint(
          repo_id=repo_id,
          temperature=0.5,
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
      template = """You are an assistant specialized in answering questions about cats. 
      If asked specificlaly about a cat available for adoption, use the retrieved context to provide accurate and concise answers. 
      If asked a general question about cats or pets in general, do not use the retrieved content.
      Politely and cutely decline to answer questions about topics unrelated to pets in general or pet adoption. 
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
