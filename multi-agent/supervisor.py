from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
model = ChatOpenAI()

def supervisor(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    # state의 중요한 부분(예: state["messages"])을 LLM에 전달해
    # 다음에 어떤 에이전트를 호출할지 결정하게 할 수 있습니다.
    # 일반적으로는 LLM에 구조화된 출력(예: "next_agent"라는 필드를 포함한 JSON 등)을 요청합니다.
    response = model.invoke(...)
    # supervisor의 결정에 따라 해당 에이전트로 이동하거나 종료합니다.
    # supervisor가 "__end__"를 반환하면 그래프의 실행이 종료됩니다.
    return Command(goto=response["next_agent"])

def agent_1(state: MessagesState) -> Command[Literal["supervisor"]]:
    # state의 필요한 정보(예: state["messages"])를 LLM에 전달하고,
    # 다른 모델, 맞춤형 프롬프트, 구조화된 출력 등 원하는 추가 로직을 적용할 수 있습니다.
    response = model.invoke(...)
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )

def agent_2(state: MessagesState) -> Command[Literal["supervisor"]]:
    # agent_2도 마찬가지로 LLM에 필요한 정보를 전달하고 응답을 받아 처리합니다.
    response = model.invoke(...)
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )

builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor)
builder.add_node("agent_1", agent_1)
builder.add_node("agent_2", agent_2)

# START 노드에서 supervisor로 먼저 이동하도록 설정
builder.add_edge(START, "supervisor")

# 그래프를 컴파일하여 supervisor(네트워크)를 생성
supervisor = builder.compile()

from docgraph.utils import show_graph
show_graph(supervisor)