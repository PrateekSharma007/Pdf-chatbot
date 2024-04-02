import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.llms import CTransformers
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.embeddings import CohereEmbeddings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
app = Flask(__name__)
CORS(app)
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
api_key = "TqIvtGkX9cfPY954K9GUNPedCCuNheHUYL2me65S"
embeddings = CohereEmbeddings(cohere_api_key=api_key)


@app.route('/hello')
def hello():
    return Response('Hello, World!', status=200)

# @app.route("/upload", methods=["POST"])
# def upload():


model_PATH = "models\llama-2-7b-chat.Q4_K_M.gguf"
n_gpu_layers = -1
n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model = LlamaCpp(
        model_path=model_PATH,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        temperature=0.5,
        # max_tokens=2000,
        n_ctx = 2000,
        f16_kv = True,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True
    )




def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks
    


def get_vector_store(text_chunks):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # model=CTransformers(model="models\llama-2-7b-chat.ggmlv3.q4_0.bin",
    #               model_type="llama",
    #               config={'max_new_tokens':512,
    #                       'temperature':0.8})  

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using LLama2")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
