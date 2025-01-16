import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
import uuid

class State(TypedDict):
    input_text: str
    output_text: str

# 콜백 핸들러 정의
class PrintCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, prompts, **kwargs):
        print("[Callback] Chain Start:", serialized)

    def on_chain_end(self, outputs, **kwargs):
        print("[Callback] Chain End:", outputs)

    def on_chain_error(self, error, **kwargs):
        print("[Callback] Chain Error:", error)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("[Callback] LLM Start:", serialized, messages)

    def on_chat_model_stream(self, token, **kwargs):
        # 토큰 단위로 출력
        print("[Callback] LLM Token:", token.content)

    def on_chat_model_end(self, response, **kwargs):
        print("[Callback] LLM End:", response)

def call_model(state: State, config):
    llm_type = config.get("configurable", {}).get("llm", "openai")

    # 콜백 핸들러 인스턴스 생성
    callback_handler = PrintCallbackHandler()

    if llm_type == "anthropic":
        model = ChatOpenAI(model="claude-v1", callbacks=[callback_handler])
    else:
        model = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback_handler])

    user_input = state["input_text"]
    messages = [{"role": "user", "content": user_input}]
    response = model.invoke(messages)

    return {"output_text": response}

# 그래프 빌드
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        "thread_id": thread_id,
        "llm": "openai"
    }
}
initial_state = {"input_text": "Hello, can you summarize AI ethics?", "output_text": ""}

async def main():
    # .astream_events()로 그래프 실행 중 이벤트를 비동기로 수신
    async for event in graph.astream_events(initial_state, config=config, version="v1"):
        kind = event["event"]
        name = event["name"]
        # 여기서 event["data"]를 확인하면 토큰 스트림이나 상태 업데이트 정보 등 상세 데이터 접근 가능
        print(f"[Event Stream] {kind}: {name}")

if __name__ == "__main__":
    asyncio.run(main())
