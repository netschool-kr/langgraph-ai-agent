from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from operator import add

class OverallState(TypedDict):
    subjects: list[str]
    # jokes 키에 add 리듀서를 지정해, 리스트를 합치는 방식으로 업데이트
    jokes: Annotated[list[str], add]

def node_a(state: OverallState):
    # node_a에서는 초기 상태를 그대로 반환한다고 가정
    # 실제로는 여기에서 subjects를 설정할 수도 있음
    return state

def continue_to_jokes(state: OverallState):
    # subjects 리스트를 순회하며 각 subject마다 "generate_joke" 노드로 상태를 분기
    # Send를 통해 각 subject에 대한 별도 실행 경로 생성
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

def generate_joke(state: dict):
    subject = state["subject"]
    joke = f"Here's a joke about {subject}: Why did the {subject} cross the road?"
    # jokes 리스트에 append 될 수 있도록 {"jokes": [joke]} 형태로 반환
    return {"jokes": [joke]}

# 초기 state에서 subjects를 받아들일 수 있도록 설정
builder = StateGraph(OverallState)
builder.add_node("node_a", node_a)
builder.add_node("generate_joke", generate_joke)

# node_a 실행 후 subjects에 따라 여러 Send를 발생시킬 조건부 엣지 설정
builder.add_conditional_edges("node_a", continue_to_jokes)

builder.add_edge(START, "node_a")
builder.add_edge("generate_joke", END)

graph = builder.compile()

# subjects를 초기 상태로 제공
initial_state = {
    "subjects": ["cat", "dog", "banana"],
    "jokes": []  # jokes 초기값 설정
}
result = graph.invoke(initial_state)
print(result)
