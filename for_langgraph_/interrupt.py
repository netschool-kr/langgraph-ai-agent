from docgraph.utils import show_graph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

class State(TypedDict):
    # 그래프 전체 상태를 정의하는 TypedDict
    # 'input': 사용자로부터 받은 입력 텍스트
    # 'user_feedback': human_feedback 노드에서 수집하는 사용자 피드백
    input: str
    user_feedback: str

def step_1(state):
    # step_1 노드: 그래프 실행 시작 후 첫 번째로 실행될 노드
    # 여기서는 단순히 "Step 1" 출력만 수행, 상태 변경 없음
    print("---Step 1---")
    pass

def human_feedback(state):
    # human_feedback 노드: 사용자에게 피드백을 요청하는 지점
    print("---human_feedback---")
    # interrupt 호출로 그래프 실행을 일시정지하고 사용자 입력을 기다림
    feedback = interrupt("Please provide feedback:")
    # 사용자가 입력한 feedback을 상태에 반영
    return {"user_feedback": feedback}

def step_3(state):
    # step_3 노드: 사용자 피드백 반영 후 실행되는 후속 처리 노드
    print("---Step 3---")
    pass

# 그래프 빌더를 이용해 노드와 엣지를 정의
builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)

# 엣지(흐름) 정의: START → step_1 → human_feedback → step_3 → END
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

# interrupt 사용을 위해 상태를 저장할 checkpointer 설정
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

show_graph(graph)

# Input
initial_input = {"input": "hello world"}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="updates"):
    print(event)
    print("\n")