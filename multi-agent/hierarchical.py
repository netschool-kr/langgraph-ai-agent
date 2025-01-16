from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

model = ChatOpenAI()

# team_1 정의 (위의 single supervisor 예시와 동일한 패턴)

def team_1_supervisor(state: MessagesState) -> Command[Literal["team_1_agent_1", "team_1_agent_2", END]]:
    response = model.invoke(...)
    return Command(goto=response["next_agent"])

def team_1_agent_1(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
    response = model.invoke(...)
    return Command(goto="team_1_supervisor", update={"messages": [response]})

def team_1_agent_2(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
    response = model.invoke(...)
    return Command(goto="team_1_supervisor", update={"messages": [response]})
from typing import Literal
from langgraph.graph import MessagesState

class Team1State(MessagesState):
    """
    Team1State 클래스.
    LLM의 응답에서 어느 에이전트로 이동할지 결정하기 위해
    next 필드를 정의합니다.
    """
    next: Literal["team_1_agent_1", "team_1_agent_2", "__end__"]
    
team_1_builder = StateGraph(Team1State)
team_1_builder.add_node("team_1_supervisor", team_1_supervisor)
team_1_builder.add_node("team_1_agent_1", team_1_agent_1)
team_1_builder.add_node("team_1_agent_2", team_1_agent_2)
team_1_builder.add_edge(START, "team_1_supervisor")
team_1_graph = team_1_builder.compile()

# team_2 정의 (위의 single supervisor 예시와 동일한 패턴)
class Team2State(MessagesState):
    # 차후에 LLM 응답에서 어떤 에이전트로 이동할지 결정할 수 있도록
    # next 필드를 정의해 둡니다.
    next: Literal["team_2_agent_1", "team_2_agent_2", "__end__"]

def team_2_supervisor(state: MessagesState) -> Command[Literal["team_2_agent_1", "team_2_agent_2", END]]:
    # team_2의 supervisor 로직을 구현할 수 있습니다.
    response = model.invoke(...)
    return Command(goto=response["next_agent"])

def team_2_agent_1(state: MessagesState) -> Command[Literal["team_2_supervisor"]]:
    response = model.invoke(...)
    return Command(goto="team_2_supervisor", update={"messages": [response]})

def team_2_agent_2(state: MessagesState) -> Command[Literal["team_2_supervisor"]]:
    response = model.invoke(...)
    return Command(goto="team_2_supervisor", update={"messages": [response]})

team_2_builder = StateGraph(Team2State)
team_2_builder.add_node("team_2_supervisor", team_2_supervisor)
team_2_builder.add_node("team_2_agent_1", team_2_agent_1)
team_2_builder.add_node("team_2_agent_2", team_2_agent_2)
team_2_builder.add_edge(START, "team_2_supervisor")

team_2_graph = team_2_builder.compile()


# 최상위 supervisor(=top_level_supervisor) 정의

builder = StateGraph(MessagesState)
def top_level_supervisor(state: MessagesState):
    # state 중 필요한 부분(예: state["messages"])을 LLM에 전달해
    # 다음에 어느 팀을 호출할지 결정합니다.
    # 일반적으로 LLM에 구조화된 출력(예: "next_team" 필드를 반환) 등을 요청합니다.
    response = model.invoke(...)
    # supervisor의 결정에 따라 특정 팀을 호출하거나 종료합니다.
    # supervisor가 "__end__"를 반환하면 그래프 실행이 종료됩니다.
    return Command(goto=response["next_team"])

builder = StateGraph(MessagesState)
builder.add_node("top_level_supervisor", top_level_supervisor)
builder.add_node("team_1_graph", team_1_graph)
builder.add_node("team_2_graph", team_2_graph)

# START 노드에서 top_level_supervisor로 이동하도록 설정
builder.add_edge(START, "top_level_supervisor")

# 최종 그래프(워크플로우)를 컴파일
graph = builder.compile()

from docgraph.utils import show_graph
show_graph(graph)