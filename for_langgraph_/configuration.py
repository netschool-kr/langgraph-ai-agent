from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
import uuid

class State(TypedDict):
    input_text: str
    output_text: str

def get_llm(llm_type: str):
    print(f"LLM 타입: {llm_type}")
    return lambda text: f"{text} (processed by {llm_type})"

def node_a(state: State, config):
    llm_type = config.get("configurable", {}).get("llm", "openai")
    llm = get_llm(llm_type)
    output = llm(state["input_text"])
    return {"output_text": output}

def node_b(state: State):
    print("결과:", state["output_text"])
    return state

# 그래프 빌드
builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)

builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

thread_id = str(uuid.uuid4())

# config에 thread_id 추가
config = {
    "configurable": {
        "llm": "anthropic",
        "thread_id": thread_id  # thread_id를 여기서 지정
    }
}
initial_state = {"input_text": "Hello, world!", "output_text": ""}

result = graph.invoke(initial_state, config=config)
print("최종 결과 상태:", result)

checkpointer = MemorySaver()
# get the latest state snapshot
config = {"configurable": {"thread_id": thread_id}}
state = graph.get_state(config)
print(state)

# get a state snapshot for a specific checkpoint_id
config = {"configurable": {"thread_id": thread_id, "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
state = graph.get_state(config)
print(state)