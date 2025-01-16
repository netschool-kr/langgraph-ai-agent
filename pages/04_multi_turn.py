import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import glob
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("multi-turn chatbot")

# 캐쉬 directory
if not os.path.exists(".cache"):
    os.mkdir(".cache")
# upload directory
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("MultiTurn Chatbot :sunglasses:")

# 대화 기록 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = {}

st.markdown("*MultiTurn Chatbot is **really** ***cool***.")


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    selected_model_name = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    # session id를 입력
    session_id = st.text_input("session id를 입력하세요", "a1234")


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# chain 생성
def create_chain(model_name="gpt-4o"):

    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.",
            ),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name="gpt-4o")

    # 일반 Chain 생성
    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return chain_with_history


# 초기화 버턴
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화기록 출력
print_messages()
# for role, message in st.session_state["messages"]:
#    st.chat_message(role).write(message)

# 사용자 입력
user_input = st.chat_input("아이디어의 분야는 뭔가요?")
warning_msg = st.empty()

if "chain" not in st.session_state:
    st.session_state["chain"] = create_chain(model_name=selected_model_name)

# 사용자 내용을 입력한 경우
if user_input:
    chain = st.session_state["chain"]
    if chain is not None:
        response = chain.stream(
            # 질문 입력
            {"question": user_input},
            # 세션 ID 기준으로 대화를 기록합니다.
            config={"configurable": {"session_id": session_id}},
        )
        # 사용자 입력
        st.chat_message("user").write(user_input)
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
