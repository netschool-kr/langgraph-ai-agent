
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class InputState(TypedDict):

    user_input: str

 

class OutputState(TypedDict):

    graph_output: str

 

class OverallState(TypedDict):

    foo: str

    user_input: str

    graph_output: str

 

class PrivateState(TypedDict):

    bar: str

 

def node_1(state: InputState) -> OverallState:

    # 입력 스키마(InputState)에서 읽고, OverallState에 해당하는 키에 기록

    return {"foo": state["user_input"] + " name"}

 

def node_2(state: OverallState) -> PrivateState:

    # OverallState에서 읽고, PrivateState에 기록

    return {"bar": state["foo"] + " is"}

 

def node_3(state: PrivateState) -> OutputState:

    # PrivateState에서 읽고, 최종적으로 OutputState에 기록

    return {"graph_output": state["bar"] + " Lance"}

 

builder = StateGraph(OverallState, input=InputState, output=OutputState)

builder.add_node("node_1", node_1)

builder.add_node("node_2", node_2)

builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")

builder.add_edge("node_1", "node_2")

builder.add_edge("node_2", "node_3")

builder.add_edge("node_3", END)

 

graph = builder.compile()

result = graph.invoke({"user_input":"My"})

print(result)  # {'graph_output': 'My name is Lance'}