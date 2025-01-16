from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
tools = load_tools(["serpapi", "llm-math"], llm=llm)  
# serpapi: 웹 검색, llm-math: 수학 계산 등 다양한 tool 사용 가능

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 예시 질의
result = agent.run("서울의 현재 기온은 몇 도인지 알려주고, 섭씨에서 화씨로 변환해줘.")
print(result)
