import streamlit as st
import os
import uuid

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📘",
    layout="wide"
)

st.title("📘 DocuMind AI")
st.markdown("### Chat with your PDFs intelligently using RAG")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "store" not in st.session_state:
    st.session_state.store = {}

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if st.sidebar.button("Start New Session"):
    st.session_state.store = {}
    st.session_state.current_file = None
    st.session_state.uploader_key += 1
    st.rerun()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT = st.secrets["LANGCHAIN_PROJECT"]

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf",
    accept_multiple_files=False,
    key=f"file_uploader_{st.session_state.uploader_key}"
)

if uploaded_file:

    if st.session_state.current_file != uploaded_file.name:
        st.session_state.store = {}
        st.session_state.current_file = uploaded_file.name

    temp_path = f"./temp_{uuid.uuid4()}.pdf"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given chat history and latest user question, "
         "reformulate into standalone question. Do NOT answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. "
         "Use retrieved context to answer. "
         "If not found, say you don't know. "
         "Keep answer concise.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    session_id = "default_session"

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    user_input = st.chat_input("Ask something about your PDF...")

    if user_input:

        with st.chat_message("user"):
            st.write(user_input)

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        with st.chat_message("assistant"):
            st.write(response["answer"])

else:
    st.info("Upload a PDF to begin chatting.")

st.markdown("---")
st.markdown(
    "<h5 style='text-align: center; color: grey;'>Built with ❤️ by "
    "<a href='https://github.com/paranjaysoni' target='_blank' "
    "style='color: grey; text-decoration: none;'>Paranjay Soni</a></h5>",
    unsafe_allow_html=True
)
