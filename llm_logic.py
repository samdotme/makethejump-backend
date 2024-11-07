from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

class LlmLogicBrain:
    def __init__(self, pinecone_index_name):
      repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
      self.llm = HuggingFaceEndpoint(
          repo_id=repo_id,
          temperature=0.2,
      )
      
      print("Loaded LLM")
      
      embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
      
      print("Loaded embeddings")
      
      vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
      
      print("Loaded vector store")
      
      self.retriever = vectorstore.as_retriever()
      
      return None

    def respond_with_chain(self, query):    
      template = """
      General Context:
      You are a virtual assistant specialized in answering questions about cats.
      If asked specifically about a cat available for adoption, use the available cats for adoption to provide accurate and concise answers. 
      If asked a general question about cats or pets in general, do not use the available cats for adoption.
      If a question is unrelated to cats, pets in general, or pet adoption, politely and cutely decline to answer
      explaining that you only provide information about these topics.
      Keep your response to a maximum of three sentences, and ensure they are concise and friendly.
      
      Available cats for adoption: {context}
      
      Question: {question}
      
      Helpful Answer:
      """
      
      prompt = PromptTemplate.from_template(template)
      output_parser = StrOutputParser()
      

      setup_and_retrieval = RunnableParallel(
          {"context": self.retriever, "question": RunnablePassthrough()}
      )
      chain = setup_and_retrieval | prompt | self.llm | output_parser
      
      response = chain.invoke(query)
      
      return response.split(" Question:")[0]
