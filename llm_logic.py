from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

class LlmLogicBrain:
    def __init__(self, pinecone_index_name):
      return None

    def respond_with_chain(self, query):     
      return "I'm not yet implemented. Don't worry, I'll be smart soon!"
