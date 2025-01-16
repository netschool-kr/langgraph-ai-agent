from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.schema import AIMessage
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langchain.tools import tool

@tool
def simple_process_tool(input_text: str) -> str:
    """
    간단한 툴 예시: 입력 텍스트를 가공해 반환
    """
    return input_text.upper() + "!!!"
from langchain.agents import initialize_agent, AgentType
from langchain import OpenAI

llm = OpenAI(temperature=0)
tools = [simple_process_tool]  # 툴 목록
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

class State(TypedDict):
    input_text: str
    output_text: str

def call_tool_node(state: State, config):
    # 1) 사용자 입력을 읽음
    user_input = state["input_text"]

    # 2) Tool 호출 (직접) - simple_process_tool
    processed = simple_process_tool(user_input)

    # 3) 결과를 state["output_text"]에 반영
    return {"output_text": processed}

# 만약 Agent를 사용한다면:
def call_agent_node(state: State, config):
    from langchain.tools import tool_manager
    # 위에서 만든 agent를 가져왔다고 가정
    # agent.run(...) 형태로 user_input을 넣어 실행
    user_input = state["input_text"]
    response = agent.run(user_input)
    return {"output_text": response}

# 그래프 구성
builder = StateGraph(State)
builder.add_node("call_tool_node", call_tool_node)
builder.add_edge(START, "call_tool_node")
builder.add_edge("call_tool_node", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

initial_state = {
    "input_text": "안녕하세요?",
    "output_text": ""
}

result = graph.invoke(initial_state, config=config)
print("최종 상태:", result)
