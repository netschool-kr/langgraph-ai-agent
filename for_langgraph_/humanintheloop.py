from docgraph.utils import show_graph
from typing import TypedDict
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

class State(TypedDict):
    """그래프 상태를 정의한다."""
    some_text: str

def work_node1(state: State):
    # work_node1에서 간단히 메시지 출력 후 상태 반환
    print("work_node1 실행!")
    return state

def human_node(state: State):
    value = interrupt(
        {
            "현재_텍스트": state["some_text"]
        }
    )
    return {
        "some_text": value
    }

def work_node2(state: State):
    # 사용자 입력에 따라 텍스트 수정 후 this node 실행
    print("work_node2 실행! 현재 텍스트:", state["some_text"])
    return state

# 그래프 구성
builder = StateGraph(State)
builder.add_node("work_node1", work_node1)
builder.add_node("human_node", human_node)
builder.add_node("work_node2", work_node2)

# 엣지 설정
builder.add_edge(START, "work_node1")
builder.add_edge("work_node1", "human_node")
# human_node 이후 흐름은 interrupt 처리 후 사용자 선택에 따라 결정
# work_node2 → END는 아래 재개 시 Command로 처리할 예정이므로
# 여기서는 "work_node2"에서 END로 가는 엣지를 연결해둠.
builder.add_edge("work_node2", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
show_graph(graph)

thread_config = {"configurable": {"thread_id": uuid.uuid4()}}
initial_input = {"some_text": "원본 텍스트"}

interrupt_occurred = False

# 첫 번째 실행
for chunk in graph.stream(initial_input, config=thread_config):
    print("출력:", chunk)
    if "__interrupt__" in chunk:
        interrupt_occurred = True

if interrupt_occurred:
    # 사용자에게 계속 여부 묻기
    user_input = input("텍스트를 수정해 work_node2로 진행하시겠습니까? (y: 계속, n: 중단) : ")

    if user_input.lower() == 'y':
        new_text = input("수정할 텍스트를 입력하세요: ")
        print("계속합니다. work_node2로 진행합니다.")
        # Command를 통해 다음 노드를 work_node2로 설정하고 상태 업데이트
        for chunk in graph.stream(Command(update={"some_text": new_text}, goto="work_node2"), config=thread_config):
            print("재개 후 출력:", chunk)
    else:
        print("중단합니다. END로 이동.")
        # Command로 END 노드로 직접 이동(종료)
        # 여기서는 goto에 END를 지정해 흐름을 종료
        for chunk in graph.stream(Command(goto=END), config=thread_config):
            print("종료:", chunk)
