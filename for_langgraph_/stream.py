from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
import asyncio

# gpt-4o-mini 모델을 사용하는 ChatOpenAI 인스턴스 생성
model = ChatOpenAI(model="gpt-4o-mini")

def call_model(state: MessagesState):
    # state['messages']를 LLM에 전달하여 응답받은 결과를 "messages" 키로 반환
    response = model.invoke(state['messages'])
    return {"messages": response}

# MessagesState를 상태로 하는 그래프(workflow) 생성
workflow = StateGraph(MessagesState)

# call_model 노드를 그래프에 추가
workflow.add_node(call_model)

# START → call_model → END 흐름을 정의
workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", END)

# 그래프 컴파일
app = workflow.compile()

# 사용자로부터의 초기 입력 메시지 설정
inputs = [{"role": "user", "content": "hi!"}]

async def main():
    # .astream_events를 통해 그래프 실행 중 발생하는 이벤트를 비동기적으로 수신
    async for event in app.astream_events({"messages": inputs}, version="v1"):
        kind = event["event"]
        # 이벤트 유형(event)과 이벤트 이름(name)을 출력
        print(f"{kind}: {event['name']}")

if __name__ == "__main__":
    # 비동기 main 함수를 실행하여 이벤트 출력
    asyncio.run(main())
