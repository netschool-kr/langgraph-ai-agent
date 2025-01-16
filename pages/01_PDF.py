import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain import hub
import glob
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging

load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("pdf-rag-chatbot")

# 캐쉬 directory
if not os.path.exists(".cache"):
    os.mkdir(".cache")
# upload directory
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("PDF 기반 QA :sunglasses:")

# 대화 기록 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# upload된 파일이 없을 경우
if "chain" not in st.session_state:
    st.session_state["chain"] = None

st.markdown("*PDF 기반 QA* is **really** ***cool***.")


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 파일이 캐쉬에 저장
@st.cache_resource(show_spinner="upload된 파일을 처리중 입니다.")
def embed_file(file):
    fcontent = file.read()
    fpath = f".cache/files/{file.name}"
    print("fpath=", fpath)
    with open(fpath, "wb") as f:
        f.write(fcontent)

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(fpath)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    selected_prompt = "prompts/pdf-rag.yaml"
    selected_model_name = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-turbo", "gpt-4o-mini"], index=0
    )


# chain 생성
def create_chain(retriever, model_name="gpt-4o"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# upload 파일
if uploaded_file:
    # 파일 upload후 retriever 생성 : 작업시간이 많이 걸림.
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model_name)
    st.session_state["chain"] = chain

# 이전 대화기록 출력
print_messages()
# for role, message in st.session_state["messages"]:
#    st.chat_message(role).write(message)

# 사용자 입력
user_input = st.chat_input("아이디어의 분야는 뭔가요?")
warning_msg = st.empty()
# 사용자 내용을 입력한 경우
if user_input:

    chain = st.session_state["chain"]
    if chain is not None:
        # 대화내용 출력
        with st.chat_message("user"):
            st.write(user_input)

        response = chain.stream(user_input)

        with st.chat_message("assistant"):
            # 빈공간에 tocken streaming 출력한다.
            container = st.empty()
            answer = ""
            for tocken in response:
                answer += tocken
                container.markdown(answer)

        # answer = chain.invoke({"question": user_input})
        # st.chat_message("ai").write(answer)

        # 대화내용 기록
        add_message("user", user_input)
        add_message("ai", answer)
    else:
        # 파일을 업로드하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 하세요.")
