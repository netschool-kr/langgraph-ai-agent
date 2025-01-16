from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import uuid

class MessagesState(TypedDict):
    messages: list[dict]  # [{role: "user"|"assistant", content: str}]

model = ChatOpenAI(model="gpt-4o-mini")
memory = ConversationBufferMemory(return_messages=True, input_key="input", output_key="output")

def call_model_with_memory(state: MessagesState, config):
    user_input = None
    # state["messages"]에서 가장 마지막 user 메시지 찾기
    for msg in reversed(state["messages"]):
        if msg["role"] == "user":
            user_input = msg["content"]
            break
    if not user_input:
        user_input = "안녕하세요!"

    # LLM 호출
    messages = [{"role": "user", "content": user_input}]
    llm_response = model.invoke(messages)

    # llm_response가 AIMessage 혹은 문자열이 아닐 수 있으니 처리
    # 경우에 따라서는 llm_response.content 로 접근 가능
    if isinstance(llm_response, AIMessage):
        # 이미 AIMessage 객체라면 그대로 사용 가능
        final_text = llm_response.content
    elif isinstance(llm_response, str):
        # 만약 문자열이라면 그대로 사용
        final_text = llm_response
    else:
        # 튜플이나 리스트면 합쳐서 문자열화 or 필요한 부분만 추출
        # 예) llm_response = [("content", "안녕하세요!"), ...] 라면
        final_text = " ".join(str(item) for item in llm_response)

    # Memory에 user_input / final_text 저장
    memory.save_context({"input": user_input}, {"output": final_text})

    # Memory에서 대화 히스토리 불러와 state에 반영
    hist = memory.load_memory_variables({})["history"]  # HumanMessage / AIMessage 리스트
    new_history = []
    for m in hist:
        if isinstance(m, HumanMessage):
            new_history.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            new_history.append({"role": "assistant", "content": m.content})
        else:
            new_history.append({"role": "system", "content": m.content})

    return {"messages": new_history}

builder = StateGraph(MessagesState)
builder.add_node("call_model_with_memory", call_model_with_memory)
builder.add_edge(START, "call_model_with_memory")
builder.add_edge("call_model_with_memory", END)
graph = builder.compile()

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}
initial_state = {
    "messages": [
        {"role": "user", "content": "안녕하세요?"}
    ]
}
result = graph.invoke(initial_state, config=config)
print("최종 결과:", result)
