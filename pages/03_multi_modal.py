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
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.models import MultiModal

load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("multi-modal image 인식")

# 캐쉬 directory
if not os.path.exists(".cache"):
    os.mkdir(".cache")
# upload directory
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("Multimodal Image 인식 :sunglasses:")

# 대화 기록 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []


st.markdown("*Multimodal Image recognition is **really** ***cool***.")


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 파일이 캐쉬에 저장
@st.cache_resource(show_spinner="upload된 image를 처리중 입니다.")
def processing_imagefile(file):
    fcontent = file.read()
    fpath = f".cache/files/{file.name}"
    with open(fpath, "wb") as f:
        f.write(fcontent)
    return fpath


# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    uploaded_file = st.file_uploader("image 업로드", type=["jpg", "jpeg", "png"])
    selected_model_name = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    system_prompt = st.text_area(
        "system prompt",
        "당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다.\n 당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다.",
        height=160,
    )


# chain 생성
def generate_answer(image_fpath, system_prompt, user_prompt, model_name="gpt-4o"):

    llm = ChatOpenAI(
        temperature=0.1,  # 창의성 (0.0 ~ 2.0)
        model_name=model_name,  # 모델명
    )
    #    system_prompt = """당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다.
    #    당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다."""
    #
    #    user_prompt = """당신에게 주어진 표는 회사의 재무제표 입니다. 흥미로운 사실을 정리하여 답변하세요."""

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )

    answer = multimodal_llm.stream(image_fpath)
    return answer


# 이전 대화기록 출력
print_messages()
# for role, message in st.session_state["messages"]:
#    st.chat_message(role).write(message)

# tab 생성
main_tab1, main_tab2 = st.tabs(["이미지", "대화내용"])
if uploaded_file:
    image_fpath = processing_imagefile(uploaded_file)
    main_tab1.image(image_fpath)

# 사용자 입력
user_input = st.chat_input("아이디어의 분야는 뭔가요?")
warning_msg = main_tab2.empty()

# 사용자 내용을 입력한 경우
if user_input:
    if uploaded_file:
        response = generate_answer(
            image_fpath,
            system_prompt=system_prompt,
            user_prompt=user_input,
            model_name=selected_model_name,
        )

        # 대화내용 출력
        with main_tab2.chat_message("user"):
            main_tab2.write(user_input)

        with main_tab2.chat_message("assistant"):
            # 빈공간에 tocken streaming 출력한다.
            container = main_tab2.empty()
            answer = ""
            for tocken in response:
                answer += tocken.content
                container.markdown(answer)

        # answer = chain.invoke({"question": user_input})
        # st.chat_message("ai").write(answer)

        # 대화내용 기록
        add_message("user", user_input)
        add_message("ai", answer)
    else:
        # image를 업로드하라는 경고 메시지 출력
        main_tab2.error("image를 업로드 하세요.")
