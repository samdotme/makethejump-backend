from huggingface_hub import HfApi
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pinecone
# from pinecone import Pinecone
# from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate

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
      
      embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
      vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
      self.retriever = vectorstore.as_retriever()

    @staticmethod
    def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

    def respond_with_chain(self, query):     
      template = """You are an assistant specialized in answering general questions about cats. 
      If asked about a cat available for adoption, use the following pieces of retrieved context to provide accurate and concise answers. 
      If asked a general question about cats or pets in general, do not use the retrieved context.
      Politely and cutely decline to answer questions about unrelated topics, such as politics. 
      Keep your answers to a maximum of three sentences and ensure they are concise.

      {context}

      Question: {question}

      Helpful Answer:"""
      custom_rag_prompt = PromptTemplate.from_template(template)

      rag_chain = (
          {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
          | custom_rag_prompt
          | self.llm
          | StrOutputParser()
      )

      return rag_chain.invoke(query)
      
      

    def respond(self, question):
      template = """"
      Answer the following question in a few sentences:
      
      {question}
      """

      prompt = PromptTemplate.from_template(template)

      
      llm_chain = prompt | self.llm
      response = llm_chain.invoke({"question": question})
      
      return response
      