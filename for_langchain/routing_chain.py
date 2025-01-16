from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 요약/번역 PromptTemplate
summary_template = PromptTemplate(
    input_variables=["text"],
    template="다음 텍스트를 요약해줘:\n{text}"
)
translate_template = PromptTemplate(
    input_variables=["text"],
    template="다음 텍스트를 영어로 번역해줘:\n{text}"
)

# 각각 LLMChain 으로 감싸기
llm = OpenAI(temperature=0)
summary_chain = LLMChain(llm=llm, prompt=summary_template)
translate_chain = LLMChain(llm=llm, prompt=translate_template)

# 딕셔너리에 담아서 필요할 때 원하는 체인을 실행
chains = {
    "summary": summary_chain,
    "translate": translate_chain,
}

# 실제 실행
res_summary = chains["summary"].run("이 문장은 한국어 텍스트입니다.")
print("요약 결과:", res_summary)

res_translate = chains["translate"].run("이 문장을 영어로 바꿔주세요.")
print("번역 결과:", res_translate)
