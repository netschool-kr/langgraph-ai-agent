from typing_extensions import TypedDict, Literal

from langgraph.types import Command

from langgraph.graph import StateGraph, START, END

 

# 상태 정의

class State(TypedDict):

    foo: str

 

def my_node(state: State) -> Command[Literal["my_other_node"]]:

    # 상태 "foo" 값을 "bar"로 업데이트하고,

    # 다음 노드로 "my_other_node"를 지정

    return Command(

        update={"foo": "bar"},

        goto="my_other_node"

    )

 

def my_other_node(state: State):

    # 여기서는 단순히 state를 반환하거나, 다른 로직을 수행할 수 있음

    print("my_other_node 실행! foo =", state["foo"])

    return state

 

# 그래프 생성

builder = StateGraph(State)

 

# 노드 추가

builder.add_node("my_node", my_node)

builder.add_node("my_other_node", my_other_node)

 

# 그래프 시작 및 종료 엣지 설정

builder.add_edge(START, "my_node")

builder.add_edge("my_other_node", END)

 

# 그래프 컴파일

graph = builder.compile()

 

# 초기 상태 설정

initial_state = {"foo": "initial"}

 

# 그래프 실행

result = graph.invoke(initial_state)

print("최종 결과:", result)