import os, getpass
import sys
# 현재 작업 디렉터리를 sys.path에 추가해 라이브러리를 import 가능하게 함
sys.path.insert(0, os.path.abspath("."))

from docgraph.utils import show_graph

def _set_env(var: str):
    # 환경변수 var가 설정되어 있지 않으면,
    # getpass를 통해 사용자에게 입력받아 os.environ에 저장
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# 필요한 환경변수를 설정
_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"

from langchain_openai import ChatOpenAI

# LLM에 전달할 프롬프트 정의
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

# LLM 설정 (GPT-4o 모델, 온도=0)
model = ChatOpenAI(model="gpt-4o", temperature=0)

import operator
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel

# LLM에서 받아올 서브토픽 구조
class Subjects(BaseModel):
    subjects: list[str]

# LLM에서 '최고의 농담' 선택 시 반환할 구조
class BestJoke(BaseModel):
    id: int

# 전체 그래프에서 사용할 상태(OverallState)
class OverallState(TypedDict):
    topic: str                # 메인 주제
    subjects: list            # 서브토픽 리스트
    jokes: Annotated[list, operator.add]    # 생성된 농담 리스트(병렬 실행 시 리스트 병합)
    best_selected_joke: str   # 최종적으로 선택된 농담

# 주제(topic)에 대한 서브토픽 생성 함수
def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    # 구조화된 출력(Subjects)을 이용해 LLM 호출
    response = model.with_structured_output(Subjects).invoke(prompt)
    # LLM이 만든 서브토픽 subjects를 반환
    return {"subjects": response.subjects}

from langgraph.constants import Send

# 서브토픽별로 농담을 만들기 위해 병렬 실행을 유도하는 함수
def continue_to_jokes(state: OverallState):
    # subjects 리스트에 들어있는 각 주제에 대해
    # 'generate_joke' 노드를 호출하도록 Send 객체 리스트를 생성
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

# 농담 생성을 위한 노드에 전달될 상태 구조
class JokeState(TypedDict):
    subject: str  # 서브토픽

# LLM에서 받은 농담 정보를 저장할 구조
class Joke(BaseModel):
    joke: str

# 실제 농담을 생성하는 노드 함수
def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    # jokes 키에 리스트 형태로 농담을 추가해 반환
    return {"jokes": [response.joke]}

# 여러 농담 중 최고의 농담을 선택하는 노드
def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    # LLM에서 선택된 농담 ID를 받아옴
    response = model.with_structured_output(BestJoke).invoke(prompt)
    # ID에 해당하는 농담을 best_selected_joke에 저장
    return {"best_selected_joke": state["jokes"][response.id]}

from IPython.display import Image
from langgraph.graph import END, StateGraph, START

# 그래프 구성
graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)

# 노드 간 연결
graph.add_edge(START, "generate_topics")
# generate_topics 후, 조건부(continue_to_jokes)로 여러 번 'generate_joke' 호출
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)

# 그래프 컴파일(실행 가능한 형태로 생성)
app = graph.compile()
show_graph(app)

# 그래프 호출: 예시로 "animals"라는 주제에 대해 서브토픽과 농담을 생성 후, 최고의 농담 선정
for s in app.stream({"topic": "animals"}):
    print(s)
