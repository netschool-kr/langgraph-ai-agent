from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain import PromptTemplate

# 1) 영어 문장을 한국어로 번역
template_ko = """다음 영어 문장을 한국어로 번역해 줘:
{english_text}"""
prompt_ko = PromptTemplate(input_variables=["english_text"], template=template_ko)
chain_ko = LLMChain(llm=OpenAI(), prompt=prompt_ko)

# 2) 번역된 문장을 다시 요약
template_summary = """다음 문장을 간단히 요약해줘:
{korean_text}"""
prompt_summary = PromptTemplate(input_variables=["korean_text"], template=template_summary)
chain_summary = LLMChain(llm=OpenAI(), prompt=prompt_summary)

# 체인 연결
overall_chain = SimpleSequentialChain(chains=[chain_ko, chain_summary])

result = overall_chain.run("I have a dream that one day this nation will rise up...")
print(result)
