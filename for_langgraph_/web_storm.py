import sys
import os
import asyncio

# 현재 디렉토리를 시스템 경로에 추가하여 필요한 모듈을 찾을 수 있도록 설정합니다.
sys.path.insert(0, os.path.abspath("."))

# 필요한 모듈을 임포트합니다.
from docgraph.utils import show_graph  # 그래프를 시각화하는 데 사용되는 유틸리티 함수
from uuid import uuid4

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"web_storm - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

from dotenv import load_dotenv  # .env 파일에서 환경 변수를 로드하기 위한 함수
from langchain_community.tools import TavilySearchResults

load_dotenv()

from langchain_openai import ChatOpenAI

fast_llm = ChatOpenAI(model="gpt-4o-mini")
long_context_llm = ChatOpenAI(model="gpt-4o")

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field

direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 위키백과 작성자 역할을 하고 있습니다. 사용자가 제공한 주제에 대해 위키백과 페이지의 개요를 작성하세요. 내용은 포괄적이고 구체적으로 작성해 주세요.",
        ),
        ("user", "{topic}"),
    ]
)


class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the Wikipedia page")
    sections: List[Section] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


generate_outline_direct = direct_gen_outline_prompt | fast_llm.with_structured_output(
    Outline
)
example_topic = (
    "백만 개 이상의 토큰 컨텍스트 윈도우를 가진 언어 모델이 RAG에 미치는 영향"
)

initial_outline = generate_outline_direct.invoke({"topic": example_topic})

print(initial_outline.as_str)

# gen_related_topics_prompt = ChatPromptTemplate.from_template(
#     """저는 아래에 언급된 주제에 대한 위키백과 페이지를 작성하고 있습니다. 이 주제와 밀접하게 관련된 주제에 대한 위키백과 페이지를 추천해 주세요. 이 주제와 일반적으로 관련 있는 흥미로운 요소에 대한 통찰을 제공하거나, 유사한 주제의 위키백과 페이지에 포함되는 전형적인 내용과 구조를 이해하는 데 도움이 될 만한 예시를 찾고 있습니다.

# 가능한 많은 주제와 URL을 나열해 주세요.

# 관심 주제: {topic}
# """
# )


# class RelatedSubjects(BaseModel):
#     topics: List[str] = Field(
#         description="관련 주제의 배경 연구를 위한 포괄적인 목록.",
#     )


# expand_chain = gen_related_topics_prompt | fast_llm.with_structured_output(
#     RelatedSubjects
# )

# print(example_topic)
# related_subjects = await expand_chain.ainvoke({"topic": example_topic})
# print(related_subjects)


class Editor(BaseModel):
    affiliation: str = Field(
        description="편집자의 주요 소속.",
    )
    name: str = Field(description="에디터의 이름.", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    role: str = Field(
        description="주제와 관련된 편집자의 역할.",
    )
    description: str = Field(
        description="편집자의 초점, 관심사, 그리고 동기에 대한 설명",
    )

    @property
    def persona(self) -> str:
        return f"이름: {self.name}\n역할: {self.role}\n소속: {self.affiliation}\n설명: {self.description}\n"


class Perspectives(BaseModel):
    editors: List[Editor] = Field(
        description="편집자들의 역할과 소속을 포함한 포괄적인 목록",
        # Add a pydantic validation/restriction to be at most M editors
    )


gen_perspectives_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """다양하고 독특한 관점을 가진 위키피디아 편집자 그룹을 선정해야 합니다. 이들은 함께 협력하여 해당 주제에 대한 포괄적인 기사를 작성할 것입니다. 각 편집자는 이 주제와 관련된 서로 다른 관점, 역할 또는 소속을 대표합니다. 관련 주제의 다른 위키피디아 페이지를 참고해도 좋습니다. 각 편집자에 대해서는 그들이 집중할 내용에 대한 설명도 추가해 주세요.""",
        ),
        ("user", "관심 주제: {topic}"),
    ]
)

gen_perspectives_chain = gen_perspectives_prompt | ChatOpenAI(
    model="gpt-4o"
).with_structured_output(Perspectives)


from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain as as_runnable


@as_runnable
async def survey_subjects(topic: str):
    # 롤 플레이을 위한 주제 전문가 생성
    return await gen_perspectives_chain.ainvoke({"topic": topic})


print(example_topic)

perspectives = asyncio.run(survey_subjects.ainvoke(example_topic))
print(perspectives.dict())

from typing import Annotated

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START


def add_messages(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right


def update_references(references, new_references):
    if not references:
        references = {}
    references.update(new_references)
    return references


def update_editor(editor, new_editor):
    # Can only set at the outset
    if not editor:
        return new_editor
    return editor


class InterviewState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    references: Annotated[Optional[dict], update_references]
    editor: Annotated[Optional[Editor], update_editor]


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder

gen_qn_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ 당신은 경험 많은 위키피디아 작성자로, 특정 페이지를 편집하고자 합니다. \
위키피디아 작성자로서의 정체성 외에도, 해당 주제를 연구할 때 특정 초점에 중점을 두고 있습니다. \
지금은 전문가와 대화하여 정보를 얻고 있습니다. 유용한 정보를 얻기 위해 좋은 질문을 하세요.

질문할 내용이 더 이상 없으면, 대화를 끝내기 위해 "Thank you so much for your help!"라고 말하세요. \
한 번에 한 가지 질문만 하고, 이전에 했던 질문은 다시 하지 마세요. \
질문은 작성하려는 주제와 관련되어야 합니다. \
포괄적이고 호기심을 가지고, 전문가로부터 가능한 한 독창적인 통찰을 많이 얻으세요.

특정 관점에 충실하세요.

{persona}""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


def tag_with_name(ai_message: AIMessage, name: str):
    ai_message.name = name
    return ai_message


def swap_roles(state: InterviewState, name: str):
    converted = []
    for message in state["messages"]:
        if isinstance(message, AIMessage) and message.name != name:
            message = HumanMessage(**message.model_dump(exclude={"type"}))
        converted.append(message)
    return {"messages": converted}


@as_runnable
async def generate_question(state: InterviewState):
    editor = state["editor"]
    gn_chain = (
        RunnableLambda(swap_roles).bind(name=editor.name)
        | gen_qn_prompt.partial(persona=editor.persona)
        | fast_llm
        | RunnableLambda(tag_with_name).bind(name=editor.name)
    )
    result = await gn_chain.ainvoke(state)
    return {"messages": [result]}


print(perspectives.editors[0])
messages = [HumanMessage(f"그래서 {example_topic}에 대한 기사를 쓰고 있다고 했나요?")]


async def get_question(editor, msg):
    question = await generate_question.ainvoke(
        {
            "editor": perspectives.editors[0],
            "messages": messages,
        }
    )
    return question


question = asyncio.run(get_question(perspectives.editors[0], messages))
print(question["messages"][0].content)

print(perspectives.editors[1])

messages = [HumanMessage(f"그래서 {example_topic}에 대한 기사를 쓰고 있다고 했나요?")]
question = asyncio.run(get_question(perspectives.editors[1], messages))
print(question["messages"][0].content)

print(perspectives.editors[3])

messages = [HumanMessage(f"그래서 {example_topic}에 대한 기사를 쓰고 있다고 했나요?")]
question = asyncio.run(get_question(perspectives.editors[3], messages))
print(question["messages"][0].content)


class Queries(BaseModel):
    queries: List[str] = Field(
        description="사용자의 질문에 답하기 위한 검색 엔진 쿼리의 포괄적인 목록.",
    )


gen_queries_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 유용한 연구 보조자입니다. 검색 엔진을 사용하여 사용자의 질문에 답하세요.",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)
gen_queries_chain = gen_queries_prompt | ChatOpenAI(
    model="gpt-4o-mini"
).with_structured_output(Queries, include_raw=True)


async def gen_queries(question):
    queries = await gen_queries_chain.ainvoke(
        {"messages": [HumanMessage(content=question["messages"][0].content)]}
    )
    return queries


queries = asyncio.run(gen_queries(question))
queries["parsed"].queries


class AnswerWithCitations(BaseModel):
    answer: str = Field(
        description="사용자의 질문에 대한 인용을 포함한 포괄적인 답변.",
    )
    cited_urls: List[str] = Field(
        description="답변에 인용된 URL 목록.",
    )

    @property
    def as_str(self) -> str:
        return f"{self.answer}\n\n인용:\n\n" + "\n".join(
            f"[{i+1}]: {url}" for i, url in enumerate(self.cited_urls)
        )


gen_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 정보를 효과적으로 활용할 수 있는 전문가입니다. 지금 당신이 알고 있는 주제에 대해 위키피디아 페이지를 작성하려는 위키피디아 작성자와 대화 중입니다. 관련 정보를 수집했으며, 이제 이 정보를 사용하여 응답을 작성할 것입니다.

응답을 가능한 한 정보성 있게 작성하고, 모든 문장은 수집된 정보에 의해 뒷받침되도록 하세요. 각 응답은 신뢰할 수 있는 출처에서 가져온 인용을 포함하여 각주로 표기되며, 응답 후 URL을 재현해 주십시오.""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

gen_answer_chain = gen_answer_prompt | fast_llm.with_structured_output(
    AnswerWithCitations, include_raw=True
).with_config(run_name="GenerateAnswer")

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool


# Tavily is typically a better search engine, but your free queries are limited
search_engine = TavilySearchResults(max_results=4)


@tool
async def search_engine(query: str):
    """Search engine to the internet."""
    results = search_engine.invoke(query)
    return [{"content": r["content"], "url": r["url"]} for r in results]


# DDG
# search_engine = DuckDuckGoSearchAPIWrapper()


# @tool
# async def search_engine(query: str):
#     """Search engine to the internet."""
#     results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)
#     return [{"content": r["body"], "url": r["href"]} for r in results]

import json

from langchain_core.runnables import RunnableConfig


async def gen_answer(
    state: InterviewState,
    config: Optional[RunnableConfig] = None,
    name: str = "Subject_Matter_Expert",
    max_str_len: int = 15000,
):
    swapped_state = swap_roles(state, name)  # Convert all other AI messages
    # 쿼리 생성
    queries = await gen_queries_chain.ainvoke(swapped_state)
    query_results = await search_engine.abatch(
        queries["parsed"].queries, config, return_exceptions=True
    )
    successful_results = [
        res for res in query_results if not isinstance(res, Exception)
    ]
    # url와 콘텐츠 추출
    all_query_results = {
        res["url"]: res["content"] for results in successful_results for res in results
    }
    # We could be more precise about handling max token length if we wanted to here
    dumped = json.dumps(all_query_results)[:max_str_len]
    ai_message: AIMessage = queries["raw"]
    tool_call = queries["raw"].tool_calls[0]
    tool_id = tool_call["id"]
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])
    # Only update the shared state with the final answer to avoid
    # polluting the dialogue history with intermediate messages
    generated = await gen_answer_chain.ainvoke(swapped_state)
    cited_urls = set(generated["parsed"].cited_urls)
    # Save the retrieved information to a the shared state for future reference
    cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
    formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
    return {"messages": [formatted_message], "references": cited_references}


example_answer = asyncio.run(
    gen_answer({"messages": [HumanMessage(content=question["messages"][0].content)]})
)
print(example_answer["messages"][-1].content)

max_num_turns = 5
from langgraph.pregel import RetryPolicy


def route_messages(state: InterviewState, name: str = "Subject_Matter_Expert"):
    messages = state["messages"]
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    if num_responses >= max_num_turns:
        return END
    last_question = messages[-2]
    if last_question.content.endswith("Thank you so much for your help!"):
        return END
    return "ask_question"


builder = StateGraph(InterviewState)

builder.add_node("ask_question", generate_question, retry=RetryPolicy(max_attempts=5))
builder.add_node("answer_question", gen_answer, retry=RetryPolicy(max_attempts=5))
builder.add_conditional_edges("answer_question", route_messages)
builder.add_edge("ask_question", "answer_question")

builder.add_edge(START, "ask_question")
interview_graph = builder.compile(checkpointer=False).with_config(
    run_name="Conduct Interviews"
)
show_graph(interview_graph)

final_step = None

initial_state = {
    "editor": perspectives.editors[0],
    "messages": [
        AIMessage(
            content=f"그래서 {example_topic}에 대한 기사를 쓰고 있다고 했나요?",
            name="Subject_Matter_Expert",
        )
    ],
}


async def main():
    async for step in interview_graph.astream(initial_state):
        name = next(iter(step))
        print(name)
        print("-- ", str(step[name]["messages"])[:300])
    final_step = step


asyncio.run(main())
