from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory

# 1) ConversationChain: 대화 맥락 유지
llm = OpenAI(temperature=0.7)
conversation_memory = ConversationBufferMemory()
conversation_chain = ConversationChain(llm=llm, memory=conversation_memory)

# 2) LLMChain: 특정 요청(번역) 처리
translate_prompt = PromptTemplate(
    input_variables=["text"],
    template="다음 텍스트를 영어로 번역해 줘:\n{text}"
)
translate_chain = LLMChain(llm=llm, prompt=translate_prompt)

# 3) LLMChain: 번역 결과를 요약
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="다음 영어 텍스트를 짧게 요약해 줘:\n{text}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# 4) SequentialChain으로 2)와 3) 연결
translate_and_summary_chain = SimpleSequentialChain(chains=[translate_chain, summary_chain])

# 시나리오:
# - 대화를 통해 번역 요청이 들어오면 해당 텍스트를 별도 체인으로 번역 & 요약
# - 번역 & 요약 결과를 최종 대화에 합류시켜 사용자에게 전달
user_input = "안녕! 이 긴 문단을 영어로 번역하고 요약해 줘."

# 1) 대화형으로 사용자의 요청 분석
conversation_reply = conversation_chain.run(user_input)
print("[ConversationChain 결과]", conversation_reply)

# 2) 사용자가 입력한 텍스트에서 핵심만 뽑아 체인에 전달한다고 가정
text_to_translate = "이 긴 문단"  # 실제로는 conversation_reply에서 추출 가능

# 3) 번역 & 요약
translated_and_summarized = translate_and_summary_chain.run(text_to_translate)
print("[번역 & 요약 결과]", translated_and_summarized)

# 4) 다시 ConversationChain으로 결과를 전달
final_reply = conversation_chain.run(f"결과를 사용자에게 알려줘: {translated_and_summarized}")
print("[최종 응답]", final_reply)
