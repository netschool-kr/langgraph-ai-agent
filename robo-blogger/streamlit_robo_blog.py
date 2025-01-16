import streamlit as st
import os
from uuid import uuid4
from dotenv import load_dotenv

# LangGraph 관련 임포트
from agent.graph import graph
from agent.state import BlogStateInput
from langchain_core.runnables import RunnableConfig

def normalize_url(url: str) -> str:
    """URL이 http:// 또는 https://로 시작하지 않으면, https://를 자동 추가."""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url

def add_url_callback():
    """버튼 클릭 시 호출될 콜백 함수."""
    user_url = st.session_state.get("temp_url", "").strip()
    if user_url:
        valid_url = normalize_url(user_url)
        st.session_state["url_list"].append(valid_url)
        # 입력값 초기화
        st.session_state["temp_url"] = ""
        st.success(f"URL이 추가되었습니다: {valid_url}")
    else:
        st.warning("추가할 URL을 입력하세요.")

def clear_url_callback():
    """URL 목록 초기화 콜백 함수."""
    st.session_state["url_list"] = []
    st.info("URL 목록이 초기화되었습니다.")

def main():
    st.set_page_config(page_title="Robo Blog", layout="wide")
    st.title("Robo Blog Demo")

    # --- 사이드바 설정 ---
    st.sidebar.title("Robo Blog Settings")
    st.sidebar.info("녹음된 파일과 참고할 URL을 입력하세요.")

    # (1) 녹음->텍스트 변환 파일 업로드
    uploaded_file = st.sidebar.file_uploader(
        label="녹음 파일 업로드 (transcribed_notes_file)",
        type=["txt", "md"],
        help="녹음된 내용을 텍스트로 변환한 파일을 업로드하세요."
    )

    # (2) URL 목록 세션 스테이트 초기화
    if "url_list" not in st.session_state:
        st.session_state["url_list"] = []

    # (3) URL 입력 (key="temp_url" 사용)
    st.sidebar.subheader("참고할 URL 목록")
    st.sidebar.text_input(
        label="URL 추가",
        placeholder="https://example.com 또는 example.com",
        key="temp_url"
    )

    col_add, col_clear = st.sidebar.columns([1, 1])

    # "추가" 버튼: on_click으로 콜백 지정
    col_add.button("추가", on_click=add_url_callback)
    # "초기화" 버튼: on_click으로 콜백 지정
    col_clear.button("초기화", on_click=clear_url_callback)

    # 현재 URL 리스트 표시
    if st.session_state["url_list"]:
        st.sidebar.write("#### 입력된 URL 목록:")
        for i, url in enumerate(st.session_state["url_list"]):
            st.sidebar.write(f"{i+1}. {url}")
    else:
        st.sidebar.write("URL이 없습니다. 추가해 주세요.")

    # (4) 블로그 생성 실행 버튼
    if st.sidebar.button("블로그 생성"):
        with st.spinner("블로그 생성 중... 잠시만 기다려주세요."):
            # --- 환경 변수 설정 ---
            unique_id = uuid4().hex[0:8]  # 고유 식별자 (처음 8자만 사용)
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = f"robo-blog - {unique_id}"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

            # 2) .env 파일 로드 (API 키 등)
            load_dotenv()

            # (5) 업로드된 파일 처리
            if uploaded_file is not None:
                transcribed_notes_path = f"transcribed_notes_{unique_id}.txt"
                with open(transcribed_notes_path, "wb") as f:
                    f.write(uploaded_file.read())
            else:
                transcribed_notes_path = "agents.txt"

            # (6) URL 목록
            urls_list = st.session_state["url_list"]

            # (7) 에이전트에 전달할 입력 데이터 구성
            input_data = BlogStateInput(
                transcribed_notes_file=transcribed_notes_path,
                urls=urls_list
            )

            # (8) 옵션 설정(RunnableConfig)
            my_custom_config = {
                "configurable": {
                    "blog_structure": """The blog post should follow this structure:
1. Introduction
2. Main Body
3. Conclusion
"""
                }
            }
            runnable_config = RunnableConfig(**my_custom_config)

            # (9) 그래프 실행
            result = graph.invoke(input_data, config=runnable_config)

            # (10) 결과 출력
            st.success("블로그 생성이 완료되었습니다!")
            final_blog = result.get('final_blog', '')
            st.text_area("Generated Blog Output", final_blog, height=300)

if __name__ == "__main__":
    main()
