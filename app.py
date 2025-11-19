import os
import shutil
import logging
import tempfile

import streamlit as st

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain.prompts import ChatPromptTemplate , PromptTemplate

st.set_page_config(
    page_title = 'Ollama PDF RAG Streamlit UI' , 
    page_icon = '🎈' , 
    layout = 'wide' , 
    initial_sidebar_state = 'collapsed'
)

logging.basicConfig(
    level = logging.INFO , 
    format = '%(asctime)s - %(levelname)s - %(message)s' , 
    datefmt = '%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)

def create_vector_db(file_upload) : 

    logger.info(f'Creating vector DB from file upload: {file_upload.name}')
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir , file_upload.name)

    with open(path , 'wb') as f : 

        f.write(file_upload.getvalue())

        logger.info(f'File saved to temporary path: {path}')

        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 100)
    chunks = text_splitter.split_documents(data)

    logger.info('Document split into chunks')

    embeddings = OllamaEmbeddings(model = 'nomic-embed-text')
    vector_db = FAISS.from_documents(
        documents = chunks , 
        embedding = embeddings , 
        # collection_name = 'myRAG' , 
        # client_settings=Settings(tenant="default_tenant", database="default_database")
    )

    logger.info('Vector DB created')

    shutil.rmtree(temp_dir)
    logger.info(f'Temporary directory {temp_dir} removed')

    return vector_db


def process_question(question , vector_db , selected_model) : 

    logger.info(f'Processing question: {question} using model: {selected_model}')

    local_model = 'llama3.2'
    llm = ChatOllama(model = local_model)

    QUERY_PROMPT = PromptTemplate(
        input_variables = ['question'] , 
        template = '''You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}'''
    )


    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever() , 
        llm , 
        prompt = QUERY_PROMPT
    )

    template = '''Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    '''

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {
                'context' : retriever , 
                'question' : RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    response = chain.invoke(question)
    
    logger.info('Question processed and response generated')
    
    return response

def delete_vector_db(vector_db) : 

    logger.info('Deleting vector DB')

    if vector_db is not None : 

        vector_db.delete_collection()

        st.session_state.pop('pdf_pages' , None)
        st.session_state.pop('file_upload' , None)
        st.session_state.pop('vector_db' , None)

        st.success('Collection and temporary files deleted successfully.')
        
        logger.info('Vector DB and related session state cleared')
        
        st.rerun()
    
    else : 
        
        st.error('No vector database found to delete.')
        logger.warning('Attempted to delete vector DB, but none was found')


def main() -> None : 

    st.subheader('🧠 Ollama PDF RAG playground' , divider = 'gray' , anchor = False)

    col1 , col2 = st.columns([1.5 , 2])

    if 'messages' not in st.session_state : st.session_state['messages'] = []
    if 'vector_db' not in st.session_state : st.session_state["vector_db"] = None
    if 'use_sample' not in st.session_state : st.session_state["use_sample"] = False
    if 'selected_model' not in st.session_state : st.session_state['selected_model'] = 'nomic-embed-text'

    selected_model = st.session_state['selected_model']

    use_sample = col1.toggle(
        'Use sample PDF (Scammer Agent Paper)' , 
        key = 'sample_checkbox'
    )

    if use_sample != st.session_state.get('use_sample') : 

        if st.session_state['vector_db'] is not None : 
            
            st.session_state['vector_db'].delete_collection()
            
            st.session_state['vector_db'] = None
            st.session_state['pdf_pages'] = None
        
        st.session_state['use_sample'] = use_sample

    if use_sample : 

        sample_path = 'scammer-agent.pdf'

        if os.path.exists(sample_path) : 

            if st.session_state['vector_db'] is None : 
                with st.spinner('Processing sample PDF...') : 

                    loader = UnstructuredPDFLoader(file_path = sample_path)
                    data = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 100)
                    chunks = text_splitter.split_documents(data)

                    st.session_state['vector_db'] = FAISS.from_documents(
                        documents = chunks , 
                        embedding = OllamaEmbeddings(model = 'nomic-embed-text') , 
                        # collection_name = 'myRAG' , 
                        # client_settings=Settings(tenant="default_tenant", database="default_database")
                    )

        else : st.error("Sample PDF file not found in the current directory.")
    else : 

        file_upload = col1.file_uploader(
            'Upload a PDF file ↓' , 
            type = 'pdf' , 
            key = 'pdf_uploader'
        )

        if file_upload : 

            if st.session_state['vector_db'] is None : 

                with st.spinner('Processing uploaded PDF...') : st.session_state["vector_db"] = create_vector_db(file_upload)

    delete_collection = col1.button(
        '⚠️ Delete collection' , 
        type = 'secondary' , 
        key = 'delete_button'
    )

    if delete_collection : delete_vector_db(st.session_state['vector_db'])

    with col2 : 
        message_container = st.container(height = 500 , border = True)

        for index , message in enumerate(st.session_state['messages']) : 

            avatar = '🤖' if message['role'] == 'assistant' else '😎'

            with message_container.chat_message(message['role'] , avatar = avatar) : st.markdown(message['content'])

        if prompt := st.chat_input('Enter a prompt here...' , key = 'chat_input') : 

            try : 

                st.session_state['messages'].append({'role' : 'user' , 'content' : prompt})

                with message_container.chat_message('user' , avatar = '😎') : st.markdown(prompt)

                with message_container.chat_message('assistant' , avatar = '🤖') : 

                    with st.spinner(':green[processing...]') : 

                        if st.session_state['vector_db'] is not None : 
                            response = process_question(
                                prompt , 
                                st.session_state['vector_db'] , 
                                selected_model
                            )
                            
                            st.markdown(response)
                        else : st.warning('Please upload a PDF file first.')

                if st.session_state['vector_db'] is not None : st.session_state['messages'].append({'role' : 'assistant' , 'content' : response})

            except Exception as e : st.error(e , icon = '⛔️') ; logger.error(f'Error processing prompt: {e}')
        
        else : 
            if st.session_state['vector_db'] is None : st.warning('Upload a PDF file or use the sample PDF to begin chat...')

if __name__ == '__main__' : main()