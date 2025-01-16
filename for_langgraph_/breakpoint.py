from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
import uuid

class State(TypedDict):
    count: int

def node_a(state: State):
    # count를 1 증가
    new_count = state["count"] + 1
    print("node_a 실행! count =", new_count)
    return {"count": new_count}

def node_b(state: State):
    # node_b에 도달하면 브레이크포인트가 작동하여 여기서 실행 일시정지
    print("node_b 실행 (브레이크포인트 지점)")
    return state

def node_c(state: State):
    # 브레이크포인트 이후 다시 실행 시 state를 확인하고 처리 가능
    new_count = state["count"] + 1
    print("node_c 실행! count =", new_count)
    return {"count": new_count}

# 그래프 빌드
builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)

builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_c")
builder.add_edge("node_c", END)

checkpointer = MemorySaver()

# compile 시 breakpoints 파라미터로 "node_b"를 브레이크포인트 지점으로 설정
graph = builder.compile(checkpointer=checkpointer, breakpoints=["node_b"])

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

initial_state = {"count": 0}

# invoke 호출 시 node_b에 도달하면 그래프 실행이 중단됨
result = graph.invoke(initial_state, config=config)
print("최종 결과 상태:", result)

# 여기서 node_b 도달 시점에서 상태를 확인하고(예: checkpointer나 ThreadReader 활용),
# 상태를 수정하거나 Command(resume=...)로 그래프를 재개할 수 있음.
# 이 예제에서는 단순히 결과를 확인하는 데 그치지만,
# 브레이크포인트를 통해 human-in-the-loop 의사결정이나 상태 조정 등이 가능.
