from typing import Any

from typing_extensions import TypedDict

 

from langgraph.graph import StateGraph, START, END

 

class State(TypedDict):

    # The operator.add reducer fn makes this append-only

    state: str

 

class ReturnNodeValue:

    def __init__(self, node_secret: str):

       self._value = node_secret

 

    def __call__(self, state: State) -> Any:

        print(f"Adding {self._value} to {state['state']}")

        return {"state": [self._value]}

 

# Add nodes


builder = StateGraph(State)

 

# Initialize each node with node_secret

builder.add_node("a", ReturnNodeValue("I'm A"))

builder.add_node("b", ReturnNodeValue("I'm B"))

builder.add_node("c", ReturnNodeValue("I'm C"))

builder.add_node("d", ReturnNodeValue("I'm D"))

 

# Flow

builder.add_edge(START, "a")

builder.add_edge("a", "b")

builder.add_edge("b", "c")

builder.add_edge("c", "d")

builder.add_edge("d", END)

graph = builder.compile()

from docgraph.utils import show_graph
show_graph(graph)