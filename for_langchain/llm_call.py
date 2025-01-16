from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
import uuid

# 상태 정의: input_text를 입력받아 output_text를 생성할 예정
class State(TypedDict):
    input_text: str
    output_text: str

def call_model(state: State, config):
    # config에서 LLM 타입 가져오기(기본값: "openai")
    llm_type = config.get("configurable", {}).get("llm", "openai")

    # llm_type에 따라 LLM 인스턴스 생성 (예: openai, anthropic, huggingface 등)
    # 여기서는 예제로 llm_type이 "anthropic"이면 Anthropic API,
    # 아니면 OpenAI 기반 ChatOpenAI를 사용한다고 가정
    if llm_type == "anthropic":
        model = ChatOpenAI(model="claude-v1")  # 예: Anthropic Claude
    else:
        model = ChatOpenAI(model="gpt-4o-mini")  # 예: OpenAI 계열 모델
    
    user_input = state["input_text"]
    # 메시지 형태로 LLM 호출
    messages = [{"role": "user", "content": user_input}]
    response = model.invoke(messages)  # LLM 호출
    
    # 결과를 output_text에 반영
    return {"output_text": response}

# 그래프 구성
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        "llm": "anthropic"  # 이 값을 "openai"로 바꾸면 다른 모델 사용 가능
    },
    "recursion_limit": 5,   # 옵션: recursion_limit 설정
    "configurable": {"thread_id": thread_id}  # thread_id 설정
}

initial_state = {"input_text": "Hello, can you summarize AI ethics?", "output_text": ""}

# 그래프 실행
result = graph.invoke(initial_state, config=config)
print("최종 결과 상태:", result)
