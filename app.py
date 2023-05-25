# from langchain.document_loaders import YoutubeLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# import streamlit as st
# from langchain.chains import LLMChain
# from dotenv import find_dotenv, load_dotenv
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# import textwrap
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap





load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_vide_url(video_url):
    loader=YoutubeLoader.from_youtube_url(video_url)
    transcript=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db= FAISS.from_documents(docs,embeddings)
    return db


def get_response_from_query(db,query,k=4):
    docs= db.similarity_search(query,k=k)
    doc_page_content=' '.join([d.page_content for d in docs])
    chat= ChatOpenAI(model_name="gpt-3.5-turbo",temperature=.2)

    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """
 
    system_message_prompt=SystemMessagePromptTemplate.from_template(template)

# Human Quesiton Prompt
    human_template="Answer the follow question: {question}"
    human_message_prompt=HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt=ChatPromptTemplate.from_messages(
        [system_message_prompt,human_message_prompt]
    )

    chain=LLMChain(llm=chat,prompt=chat_prompt)

    response=chain.run(question=query,docs=doc_page_content)
    response=response.replace("\n","")
    return response,docs




    
st.title("Youtube Video Assistant")
st.write("This app will help you create a database of information from a YouTube video and then answer questions about the video")
video_url = st.text_input("Enter a YouTube Video URL")
video_button = st.button("Video Summary")

if video_button:
    db = create_db_from_youtube_vide_url(video_url)
    query = "What is the video about?"
    response, docs = get_response_from_query(db, query)
    print(textwrap.fill(response, width=50))
    st.write(textwrap.fill(response, width=50))


   






