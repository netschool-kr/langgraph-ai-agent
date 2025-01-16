from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent

model = ChatOpenAI()

# 이 함수는 도구로 호출될 에이전트 함수입니다.
# InjectedState 어노테이션을 사용해 state를 도구에 전달할 수 있습니다.
def agent_1(state: Annotated[dict, InjectedState]):
    """
    agent_1 함수: state 정보를 받아 적절히 처리 후 결과 문자열을 반환합니다.
    """
    # state의 필요한 부분(예: state["messages"])을 LLM에 전달하고,
    # 다른 모델이나 맞춤형 프롬프트, 구조화된 출력 등을 추가로 적용할 수 있습니다.
    response = model.invoke(...)
    # LLM 응답을 문자열 형태로 반환합니다 (도구 응답 포맷으로 사용됨).
    # create_react_agent(=supervisor)에 의해 자동으로 ToolMessage로 변환됩니다.
    return response.content

def agent_2(state: Annotated[dict, InjectedState]):
    """
    agent_2 함수: state 정보를 받아 적절히 처리 후 결과 문자열을 반환합니다.
    """
    # agent_2도 같은 방식으로 LLM을 호출해 응답을 반환합니다.
    response = model.invoke(...)
    return response.content

tools = [agent_1, agent_2]
# 툴을 호출하는 supervisor를 만드는 가장 간단한 방법은
# 사전 구성된 ReAct 에이전트 그래프를 사용하는 것입니다.
# 이 그래프는 툴을 호출하는 LLM 노드(=supervisor)와
# 툴을 실제로 실행하는 노드로 이루어져 있습니다.
supervisor = create_react_agent(model, tools)


from docgraph.utils import show_graph
show_graph(supervisor)