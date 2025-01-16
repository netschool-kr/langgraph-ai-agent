import sys
import os
from typing import Annotated, TypedDict

# langgraph의 StateGraph, START(시작점), END(종료점) 임포트
from langgraph.graph import StateGraph, START, END

# langgraph에서 메시지를 관리하기 위한 함수
from langgraph.graph.message import add_messages

# langchain_openai: ChatOpenAI 모델 사용 (OpenAI Chat)
from langchain_openai import ChatOpenAI

# dotenv: .env 파일에서 환경 변수를 로드하기 위해 사용
from dotenv import load_dotenv

# UUID 생성 (고유 ID를 위해 사용)
from uuid import uuid4


# 1) 현재 디렉토리를 시스템 경로에 추가
#    - 이 작업을 통해, 현 디렉토리에 있는 모듈을 임포트 가능하게 함
sys.path.insert(0, os.path.abspath("."))

# 2) 고유 식별자(unique_id) 생성 (처음 8자만 사용)
unique_id = uuid4().hex[0:8]

# 3) LangChain 트레이싱 기능 관련 환경 변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"LangSmith4LangGraph - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# 다시 dotenv 임포트 (중복이지만, 혹시를 대비해)
from dotenv import load_dotenv

# .env 파일 로드 (API 키 등 환경 변수 설정)
load_dotenv()


# 메시지 상태 구조를 정의하기 위한 TypedDict
class State(TypedDict):
    # messages 필드는 list 형태이며, add_messages 데코레이터를 통해 메시지 처리
    messages: Annotated[list, add_messages]


# ChatOpenAI 모델 정의
# - model="gpt-4o-mini": 실제로 존재하는 모델인지 여부는 주의 (예시용)
# - temperature=0: 출력의 일관성을 위해 온도를 0으로 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 챗봇 노드(함수) 정의
def chatbot(state: State):
    """
    state["messages"]에 담긴 대화 이력을 OpenAI 모델에 전달하고,
    응답(response)을 받아 다시 messages에 추가하여 반환한다.
    """
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


# StateGraph(상태 그래프) 생성
graph_builder = StateGraph(State)

# "chatbot" 노드를 그래프에 추가 (위에서 정의한 chatbot 함수를 노드로 사용)
graph_builder.add_node("chatbot", chatbot)

# 그래프의 시작점(START)에서 "chatbot" 노드로 연결
graph_builder.add_edge(START, "chatbot")

# "chatbot" 노드에서 END(종료점)로 연결
graph_builder.add_edge("chatbot", END)

# 그래프 컴파일(빌드 완성)
graph = graph_builder.compile()

# 초기 상태 정의
# 여기서는 사용자 역할(user)로 "제주주의 유명한 맛집 7개" 질문을 넣는다.
initial_state = {
    "messages": [{"role": "user", "content": "제주의 유명한 맛집  7개 추천해줘"}]
}

# docgraph.utils 내에 있는 show_graph 함수를 임포트해서 그래프 구조 시각화
from docgraph.utils import show_graph
show_graph(graph)

# 그래프 실행: stream() 메서드로 상태 변화를 이벤트 형태로 받아옴
for event in graph.stream(initial_state):
    # event.values() 내부에는 각 노드에서 반환된 결과 상태가 들어 있음
    for value in event.values():
        # messages의 마지막 메시지(assistant 응답)을 콘솔에 출력
        print("Assistant:", value["messages"][-1].content)
