import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain import hub
import glob
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from retriever import create_retriever
from langchain_community.chat_models import ChatOllama

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

st.title("Local RAG 기반 QA :sunglasses:")

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
    return create_retriever(fpath)


# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    selected_prompt = "prompts/pdf-rag.yaml"
    selected_model_name = st.selectbox(
        "LLM 선택", ["xionic", "ollama", "gpt-4o-mini"], index=0
    )


def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


# chain 생성
def create_chain(retriever, model_name="xionic"):
    if model_name == "xionic":
        # 단계 6: 프롬프트 생성(Create Prompt)
        # 프롬프트를 생성합니다.
        prompt = load_prompt("prompts/pdf-rag-xionic.yaml", encoding="utf-8")

        # 단계 7: 언어모델(LLM) 생성
        # 모델(LLM) 을 생성합니다.
        # llm = ChatOpenAI(model_name=model_name, temperature=0)
        llm = ChatOpenAI(
            model_name="llama-3.1-xionic-ko-70b",
            base_url="http://sionic.tech:28000/v1",
            api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
        )
    elif model_name == "ollama":
        prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")
        llm = ChatOllama(model="EEVE-Korean-10.8B:latest", temperature=0)
    else:
        prompt = load_prompt("prompts/pdf-rag-ollama.yaml", encoding="utf-8")
        llm = ChatOpenAI(model_name=model_name, temperature=0)
    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever | format_doc, "question": RunnablePassthrough()}
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
