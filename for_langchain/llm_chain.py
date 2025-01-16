from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

template = """다음 문장을 요약해줘:
{input_text}"""

prompt = PromptTemplate(
    input_variables=["input_text"],
    template=template
)

llm = OpenAI(temperature=0.7)  # 예시: OpenAI API Key 필요
chain = LLMChain(prompt=prompt, llm=llm)

summary = chain.run("여기에 요약하고 싶은 긴 텍스트를 넣으세요.")
print(summary)
