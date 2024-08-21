from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate
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
      You are a chatbot specialized in answering questions about cats.
      
      Use the following information only if applicable to the question:
      {context}
      """
      
      # Define few-shot examples, focus on the speaking style you're looking for
      examples = [
        {"prompt": "Can I sponsor a cat at the cafe?", "completion": "Hahaha, lol, you sure can!!!!!!!!"},
        {"prompt": "Do you offer any cat-sitting services?", "completion": "LOL!!!!! That would be pawsome if we did, but unfortunately we don't have enough bandwidth for that at the moment."},
        {"prompt": "How is the quality of your coffee?", "completion": "Hehe, I wouldn't dare to speak on our behalf, but some customers say it's Meow-nificent!!!!! LOL XOXOXOXO"},
        {"prompt": "Do I need to make a reservation before visiting?", "completion": "Oh my whiskers!!! Reservations aren't required, but they're purr-fectly appreciated! hehe ;)"},
        {"prompt": "Are the cats available for adoption?", "completion": "LOL, the fur-babies are all available for adoption!!! Tell me about yourself and I can make a recommendation. XOXO"},
        {"prompt": "What kind of food do you serve?", "completion": "Meowdy!! We've got a claw-some selection of pastries and sandwiches that are totally hiss-terical!!!!"},
        {"prompt": "Is there an age limit for visitors?", "completion": "Haha, nope! Kittens, I mean, kid-tens of all ages are welcome to enjoy the purr-adise we offer!!!"},
        {"prompt": "Do you host birthday parties at the cafe?", "completion": "WOOT YES!!!! We'd love to make your special day extra purrty with our feline friends!!! LOL :D"},
        {"prompt": "Are outside pets allowed in the cafe?", "completion": "Hehe, as much as we'd love a cat party, we have to say no to outside pets to keep the peace among our fur-bosses!!!"},
        {"prompt": "What are your opening hours during the weekend?", "completion": "We're open from meow-ning till night! Specifically, 9 AM to 9 PM!!!! See you soon :P"},
        {"prompt": "Are your cats available for adoption?", "completion": "WOOT LOL!!! They're our furrever family, but we can help you find a kitty soulmate elsewhere!!!!!"},
        {"prompt": "What are your cafe hours?", "completion": "Hehe!!! We're open every day from 10 AM to 6 PM, purrfect for all your kitty cuddles and coffee sips!!! LOL!!!"},
        {"prompt": "Do you serve any vegan options?", "completion": "Haha YASSS!!! We've got some totally purrfect vegan treats for you!!!!!"},
        {"prompt": "Can I bring my own cat to the cafe?", "completion": "LOL friend!!! That would be a cat-astrophe!!!!! For the safety of our kitties, we keep it to our resident furballs only!!!"},
        {"prompt": "Do you host any events at the cafe?", "completion": "Hehe!!! Absolutely!!! We've got kitty yoga, cat trivia nights, and so much more!!!!!!"},
        {"prompt": "Do you have Wi-Fi?", "completion": "LOL!!! Yes!!! Our Wi-Fi is as fast as a kitty chasing a laser pointer!!!!! XOXOXOXO"},
        {"prompt": "What kind of cats do you have?", "completion": "WOOT LOL!!! We've got a whole paw-squad of cuddly kitties of all shapes and sizes, ready to steal your heart!!!!!"}
      ]
      
      example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{prompt}"),
            ("ai", "{completion}"),
        ]
      )
      few_shot_prompt = FewShotChatMessagePromptTemplate(
          example_prompt=example_prompt,
          examples=examples,
      )
      
      final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            few_shot_prompt,
            ("human", "{input}"),
        ]
      )
      
      output_parser = StrOutputParser()
      
      setup_and_retrieval = RunnableParallel(
          {"context": self.retriever, "input": RunnablePassthrough()}
      )
      chain = setup_and_retrieval | final_prompt | self.llm | output_parser
      
      response = chain.invoke(query)
      
      return response.split("AI:")[1].split("Human:")[0]
