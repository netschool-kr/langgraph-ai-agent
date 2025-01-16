from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

model = ChatOpenAI()

def agent_1(state: MessagesState) -> Command[Literal["agent_2", "agent_3", END]]:
    # state 안에서 필요한 정보(예: state["messages"])를 LLM에 전달해서
    # 다음에 어떤 에이전트를 호출할지 결정하도록 할 수 있습니다.
    # 일반적으로 LLM에 구조화된 출력(예: "next_agent" 필드를 가진 JSON 등)을 요청합니다.
    response = model.invoke(...)
    # LLM의 결정에 따라 다른 에이전트로 이동하거나 종료할 수 있습니다.
    # LLM이 "__end__"를 반환하면 실행이 종료됩니다.
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

def agent_2(state: MessagesState) -> Command[Literal["agent_1", "agent_3", END]]:
    # agent_2 역시 LLM의 응답에 따라 다음 에이전트를 결정하고 메시지를 업데이트합니다.
    response = model.invoke(...)
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

def agent_3(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    # agent_3도 비슷한 로직으로 작동합니다.
    response = model.invoke(...)
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

builder = StateGraph(MessagesState)
builder.add_node("agent_1", agent_1)
builder.add_node("agent_2", agent_2)
builder.add_node("agent_3", agent_3)

# START 노드에서 agent_1로 먼저 이동하도록 설정합니다.
builder.add_edge(START, "agent_1")

# 구성된 그래프를 컴파일하여 네트워크(워크플로우)를 생성합니다.
network = builder.compile()

from docgraph.utils import show_graph
show_graph(network)