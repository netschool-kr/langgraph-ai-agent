import os

from agent.graph import graph
from agent.state import BlogStateInput
from langchain_core.runnables import RunnableConfig
from uuid import uuid4


# 2) 고유 식별자(unique_id) 생성 (처음 8자만 사용)
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"robo-blog - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

from dotenv import load_dotenv

# .env 파일 로드 (API 키 등 환경 변수 설정)
load_dotenv()
# (1) 환경변수 설정 (예: Anthropic 키)
#os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_KEY"

# (2) 입력 데이터 설정
input_data = BlogStateInput(
    transcribed_notes_file="agents.txt",
    urls=[
        "https://langchain-ai.github.io/langgraph/concepts/",
        "https://www.deeplearning.ai/the-batch/issue-253/"
    ]
)

# (3) 옵션 설정(RunnableConfig)
my_custom_config = {
    "configurable": {
        "blog_structure": """The blog post should follow this structure:
1. Introduction
2. Main Body
3. Conclusion
"""
    }
}
runnable_config = RunnableConfig(**my_custom_config)

# (4) 그래프 실행: run() 대신 eval() 사용
result = graph.invoke(input_data, config=runnable_config)

# (5) 최종 결과 출력
print("=== Final Blog Output ===")
print(result['final_blog'])
