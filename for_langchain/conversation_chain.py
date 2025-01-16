from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0.7)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# 사용자 발화 1
reply1 = conversation.run("안녕? 너는 누구야?")
print(reply1)

# 사용자 발화 2(이전 문맥이 유지됨)
reply2 = conversation.run("아까 이야기한 내용 요약해줄래?")
print(reply2)
