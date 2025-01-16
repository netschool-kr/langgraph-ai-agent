import os, getpass
import sys
# 현재 작업 디렉터리를 sys.path에 추가해 라이브러리를 import 가능하게 함
sys.path.insert(0, os.path.abspath("."))

from docgraph.utils import show_graph

def _set_env(var: str):
    # 만약 환경변수 var가 설정되어 있지 않다면, getpass로 입력받아서 환경변수로 설정
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# LANGCHAIN_API_KEY라는 환경변수 설정 (입력 요청)
_set_env("LANGCHAIN_API_KEY")

# LangChain 추적 관련 환경변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"

from operator import add
from typing_extensions import TypedDict
from typing import List, Optional, Annotated

# 로그 구조를 정의하는 TypedDict
class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]

from langgraph.graph import StateGraph, START, END

# --------------------
# 1) 장애(Failure) 분석을 위한 서브 그래프
# --------------------
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]    # 전처리된 로그 목록
    failures: List[Log]        # 장애가 있는 로그 목록
    fa_summary: str            # 장애 요약 정보
    processed_logs: List[str]  # 처리된 로그에 대한 식별 정보

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]

def get_failures(state):
    """ 장애가 있는 로그를 추출하는 함수 """
    cleaned_logs = state["cleaned_logs"]
    # cleaned_logs 중에서 "grade" 키가 있는 로그(장애로 가정)를 모음
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}

def generate_summary(state):
    """ 장애 요약 정보를 생성하는 함수 """
    failures = state["failures"]
    # 실제로는 요약 로직이 들어간다고 가정. 여기서는 예시 텍스트로 대체
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {
        "fa_summary": fa_summary,
        "processed_logs": [f"failure-analysis-on-log-{failure['id']}" for failure in failures]
    }

# Failure Analysis 서브 그래프를 구성
fa_builder = StateGraph(input=FailureAnalysisState, output=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)

# 그래프 상에서 노드의 흐름 정의
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)

# 최종 그래프 컴파일
graph = fa_builder.compile()

# 그래프 시각화
show_graph(graph)

# --------------------
# 2) 질문 요약(Question Summarization) 서브 그래프
# --------------------
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]    # 전처리된 로그 목록
    qs_summary: str            # 질문 요약
    report: str                # 보고서
    processed_logs: List[str]  # 처리된 로그 식별 정보

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]

def generate_summary(state):
    # cleaned_logs를 입력으로 받음
    cleaned_logs = state["cleaned_logs"]
    # 실제 요약 로직 대신 예시 텍스트로 대체
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {
        "qs_summary": summary,
        "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]
    }

def send_to_slack(state):
    # qs_summary(질문 요약) 를 받아와서 보고서(report) 생성
    qs_summary = state["qs_summary"]
    report = "foo bar baz"  # 실제 보고서 생성 로직 대신 예시
    return {"report": report}

# Question Summarization 서브 그래프 구성
qs_builder = StateGraph(input=QuestionSummarizationState, output=QuestionSummarizationOutputState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)

# 노드 흐름 정의
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)

# 그래프 컴파일
graph = qs_builder.compile()

# 그래프 시각화
show_graph(graph)

# --------------------
# 3) 상위(Entry) 그래프 구성: clean_logs -> (failure_analysis & question_summarization)
# --------------------

# 상위 그래프에서 다룰 상태 구조 정의
class EntryGraphState(TypedDict):
    raw_logs: List[Log]                      # 원본 로그
    cleaned_logs: List[Log]                  # 전처리된 로그
    fa_summary: str                          # 장애 분석 서브 그래프에서 생성
    report: str                              # 질문 요약 서브 그래프에서 생성
    processed_logs:  Annotated[List[int], add]  # 두 서브 그래프에서 모두 추가될 수 있음

def clean_logs(state):
    # raw_logs를 받아서 cleaned_logs로 전처리
    raw_logs = state["raw_logs"]
    cleaned_logs = raw_logs  # 실제론 전처리 로직이 들어갈 수 있음
    return {"cleaned_logs": cleaned_logs}

entry_builder = StateGraph(EntryGraphState)

# 상위 그래프에 노드(함수 or 서브 그래프) 등록
entry_builder.add_node("clean_logs", clean_logs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

# 노드 간 연결
entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

# 최종 상위 그래프 컴파일
graph = entry_builder.compile()

# 시각화
show_graph(graph)

# --------------------
# 4) 테스트: 더미(dummy) 로그 데이터
# --------------------
question_answer = Log(
    id="1",
    question="How can I import ChatOllama?",
    answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
)

question_answer_feedback = Log(
    id="2",
    question="How can I use Chroma vector store?",
    answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
    grade=0,  # 장애 점수(가정)
    grader="Document Relevance Recall",
    feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
)

# raw_logs 예시 목록
raw_logs = [question_answer, question_answer_feedback]

# 그래프 실행(invoke) 및 결과 출력
result = graph.invoke({"raw_logs": raw_logs})
print(result)
