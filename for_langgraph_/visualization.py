from docgraph.utils import show_graph
import random
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # 상태: messages 키에 add_messages 리듀서를 적용한 리스트
    # 이는 메시지 히스토리를 상태로 관리하기 위함
    messages: Annotated[list, add_messages]


class MyNode:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, state: State):
        # 노드 호출 시 'messages'에 새로운 메시지("assistant", "Called node ...") 추가
        return {"messages": [("assistant", f"Called node {self.name}")]} 


def route(state) -> Literal["entry_node", "__end__"]:
    # messages 길이가 10을 초과하면 __end__로,
    # 그렇지 않으면 "entry_node"로 라우팅하는 함수
    if len(state["messages"]) > 10:
        return "__end__"
    return "entry_node"


def add_fractal_nodes(builder, current_node, level, max_level):
    # 재귀적으로 노드를 추가하여 '프랙탈' 형태의 그래프를 생성하는 함수
    # level이 max_level을 초과하면 더 이상 노드 추가하지 않음
    if level > max_level:
        return

    # 이 레벨에서 생성할 노드 수(1~3개 사이 무작위)
    num_nodes = random.randint(1, 3)
    for i in range(num_nodes):
        nm = ["A", "B", "C"][i]  # 노드 이름에 사용할 suffix
        node_name = f"node_{current_node}_{nm}"
        # 현재 노드에서 새로 만든 노드로 엣지 연결
        builder.add_node(node_name, MyNode(node_name))
        builder.add_edge(current_node, node_name)

        # 다음 레벨 노드를 재귀적으로 추가하거나 조건부 라우팅/끝 노드로 연결
        r = random.random()
        if r > 0.2 and level + 1 < max_level:
            # 확률적으로 다음 레벨로 내려가 서브노드 생성
            add_fractal_nodes(builder, node_name, level + 1, max_level)
        elif r > 0.05:
            # 약간의 확률로 route 함수를 통한 조건부 라우팅 추가
            builder.add_conditional_edges(node_name, route, node_name)
        else:
            # 그 외의 경우 __end__로 바로 연결(종료)
            builder.add_edge(node_name, "__end__")


def build_fractal_graph(max_level: int):
    # max_level 깊이까지 프랙탈 형태의 노드를 생성하는 그래프 빌드
    builder = StateGraph(State)
    entry_point = "entry_node"
    builder.add_node(entry_point, MyNode(entry_point))
    builder.add_edge(START, entry_point)

    # entry_node로부터 시작해 프랙탈 노드 생성
    add_fractal_nodes(builder, entry_point, 1, max_level)

    # 필요하다면 entry_point에서 END로 연결(옵션)
    builder.add_edge(entry_point, END)

    return builder.compile()


app = build_fractal_graph(3)
# 생성된 그래프를 시각화
show_graph(app)
