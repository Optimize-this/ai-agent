import os
import streamlit as st
import tiktoken

from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain.chat_models.gigachat import GigaChat
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from dotenv import load_dotenv

load_dotenv() 

template = """Отвечай на вопрос только на основе контекста:
{context}

Вопрос: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = GigaChat(
    credentials=os.getenv("CREDENTIALS"),
    scope='GIGACHAT_API_CORP',
    verify_ssl_certs=False,
)
output_parser = StrOutputParser()

embedding = GigaChatEmbeddings(
    credentials=os.getenv("CREDENTIALS"),
    scope='GIGACHAT_API_CORP',
    verify_ssl_certs=False
)

# Sidebar contents
with st.sidebar:
    st.title('DEEPHACK.AGENTS')
    st.markdown('''
    ## О проекте
    ............
    - .......
    - .......
    - .......
    ''')
    add_vertical_space(5)
    st.write('....................')

def split_large_text(large_text, max_tokens):
    enc = tiktoken.get_encoding("cl100k_base")
    tokenized_text = enc.encode(large_text)

    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokenized_text:
        current_chunk.append(token)
        current_length += 1

        if current_length >= max_tokens:
            chunks.append(enc.decode(current_chunk).rstrip(' .,;'))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(enc.decode(current_chunk).rstrip(' .,;'))

    return chunks

def create_vector_store(data):
    docs = split_large_text(data, 450)
    vectorstore = DocArrayInMemorySearch.from_texts(docs, embedding)

    return vectorstore

def main():
    st.header("Ассистент")

    pdf = st.file_uploader("Документ для анализа", type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # st.write(f'{text}')

        vectorstore = create_vector_store(text)
        retriever = vectorstore.as_retriever()

        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )
        chain = setup_and_retrieval | prompt | model | output_parser
        
        # store_name = pdf.name[:-4]
        # st.write(f'{store_name}')

        query = st.text_input("Запрос:")

        if query:
            answer = chain.invoke(query)
            st.write(answer)


if __name__ == '__main__':
    main()