import sys
import os
from uuid import uuid4

from dotenv import load_dotenv
# LangChain의 ChatOpenAI 모델 사용 (OpenAI Chat API)
from langchain.chat_models import ChatOpenAI
# LangChain 메시지 형식 (사람 메시지, AI 메시지 등)
from langchain.schema import HumanMessage, AIMessage

# 1) 환경 설정
#    - 현재 디렉터리를 우선 탐색 경로에 추가 (필요 시)
sys.path.insert(0, os.path.abspath("."))

# 2) LangChain 트레이싱 기능 관련 환경 변수 (선택 사항)
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"LangChain-Chatbot - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# .env 파일에서 환경 변수를 로드 (예: OPENAI_API_KEY 등)
load_dotenv()

########################################
# 3) LLM 초기화
########################################
# - model_name: 실제 사용 가능한 OpenAI 모델 명 (예: "gpt-3.5-turbo", "gpt-4")
# - temperature=0: 응답의 일관성을 높이기 위해 설정 (임의 조정 가능)
# - openai_api_key: .env에 저장된 키를 가져와 사용
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

########################################
# 4) 초기 메시지 정의
########################################
# - HumanMessage를 통해 "사용자"가 묻는 내용 입력
messages = [
    HumanMessage(content="제주의 유명한 맛집 7개 추천해줘")
]

########################################
# 5) LLM 호출 (한 번의 질의/응답)
########################################
# - messages 리스트를 LLM에게 넘기면, 마지막 메시지(사람 메시지)를 기반으로 응답 생성
response = llm(messages)

########################################
# 6) 결과 출력
########################################
# - response는 AIMessage 객체이며, .content에 텍스트가 담겨 있음
print("Assistant:", response.content)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    messages.append(HumanMessage(content=user_input))
    response = llm(messages)
    messages.append(response)
    print("Assistant:", response.content)
