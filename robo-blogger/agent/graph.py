import os
from langchain_anthropic import ChatAnthropic  # Anthropic 기반 챗봇 모델 사용을 위한 모듈
from langchain_core.messages import HumanMessage, SystemMessage  # 메시지 구조 정의
from langchain_core.runnables import RunnableConfig  # 실행 설정 정보를 담는 객체

from langgraph.constants import Send  # 상태 머신 흐름 제어용 상수
from langgraph.graph import START, END, StateGraph  # 상태 머신 구성 요소(시작, 끝, 그래프)

import agent.configuration as configuration  # 설정 관련 모듈
from agent.state import Sections, BlogState, BlogStateInput, BlogStateOutput, SectionState  # 상태 정의
from agent.prompts import blog_planner_instructions, main_body_section_writer_instructions, intro_conclusion_instructions  # 프롬프트 템플릿
from agent.utils import load_and_format_urls, read_dictation_file, format_sections  # 유틸 함수
import matplotlib.pyplot as plt
from PIL import Image

def show_graph(graph, title="graph", display=True, output_path="graph.png"):
    # graph 이미지를 바이트 객체로 생성하고 파일로 저장
    png_data = graph.get_graph(xray=True).draw_mermaid_png()  # 이미지를 바이너리로 생성
    with open(output_path, "wb") as f:
        f.write(png_data)  # 이미지를 파일로 저장

    print("Graph image saved as '{output_path}'. Displaying the image...")

    # display=True일 때만 화면에 표시
    if display:
        try:
            # 저장된 이미지를 읽어서 matplotlib으로 표시
            image = Image.open("graph.png")
            plt.imshow(image)
            plt.axis("off")  # 축 숨기기
            # 현재 figure의 canvas 매니저를 통해 윈도우 타이틀 설정
            plt.gcf().canvas.manager.set_window_title(title)            
            plt.show()
        except Exception as e:
            print("그래프 표시 과정에서 에러가 발생했습니다:", e)

# ------------------------------------------------------------
# LLMs
claude_3_5_sonnet = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0, anthropic_api_key=os.environ["ANTHROPIC_API_KEY"]) 
# Anthropic의 claude-3-5-sonnet 모델을 사용, 온도(창의성)는 0으로 설정

# ------------------------------------------------------------
# Graph
def generate_blog_plan(state: BlogState, config: RunnableConfig):
    """ 블로그 계획(섹션 구성) 생성 함수 """

    # 1) 음성 인식 파일에서 텍스트를 읽어옴
    user_instructions = read_dictation_file(state.transcribed_notes_file)

    # 2) RunnableConfig에서 블로그 구조 설정 불러오기
    configurable = configuration.Configuration.from_runnable_config(config)
    blog_structure = configurable.blog_structure

    # 3) 블로그 계획 작성에 필요한 시스템 지시사항 생성
    system_instructions_sections = blog_planner_instructions.format(
        user_instructions=user_instructions, 
        blog_structure=blog_structure
    )

    # 4) LLM 호출: 섹션 목록(Sections)을 구조화된 형태로 생성
    structured_llm = claude_3_5_sonnet.with_structured_output(Sections,
                                                              anthropic_api_key=os.environ["ANTHROPIC_API_KEY"] 
                                                              )
    report_sections = structured_llm.invoke([
        SystemMessage(content=system_instructions_sections),
        HumanMessage(content="Generate the sections of the blog. Your response must include a 'sections' field containing a list of sections. Each section must have: name, description, and content fields.")
    ])

    # 5) 생성된 섹션 목록 반환
    return {"sections": report_sections.sections}

def write_section(state: SectionState):
    """ 본문 섹션을 작성하는 함수 """

    # 1) 현재 섹션 정보와 참고 URL 가져오기
    section = state.section
    urls = state.urls

    # 2) 음성 인식 파일에서 텍스트를 읽어옴
    user_instructions = read_dictation_file(state.transcribed_notes_file)

    # 3) 참고할 URL들을 포맷팅하여 문자열로 변환
    url_source_str = "" if not urls else load_and_format_urls(urls)

    # 4) 시스템 지시사항 생성 (본문 섹션 작성용 프롬프트)
    system_instructions = main_body_section_writer_instructions.format(
        section_name=section.name, 
        section_topic=section.description, 
        user_instructions=user_instructions, 
        source_urls=url_source_str
    )

    # 5) LLM 호출: 섹션 내용을 생성하고, 해당 내용을 section 객체에 저장
    section_content = claude_3_5_sonnet.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate a blog section based on the provided information.")
    ])
    section.content = section_content.content

    # 6) 작성이 끝난 섹션을 반환(리스트 형태)
    return {"completed_sections": [section]}

def write_final_sections(state: SectionState):
    """ 서론, 결론 등 웹 검색이 필요 없는 섹션을 작성하는 함수 """

    # 1) 현재 섹션 정보 가져오기
    section = state.section
    
    # 2) 지시사항 생성 (서론, 결론 등 최종 섹션 작성용 프롬프트)
    system_instructions = intro_conclusion_instructions.format(
        section_name=section.name, 
        section_topic=section.description, 
        main_body_sections=state.blog_main_body_sections, 
        source_urls=state.urls
    )

    # 3) LLM 호출: 섹션 내용 작성
    section_content = claude_3_5_sonnet.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate an intro/conclusion section based on the provided main body sections.")
    ])
    section.content = section_content.content

    # 4) 작성이 끝난 섹션을 반환
    return {"completed_sections": [section]}

def initiate_section_writing(state: BlogState):
    """ 본문 섹션 작성을 위해 병렬로 작업을 시작하는 'map' 단계 """
    
    # 1) 본문 작성이 필요한 섹션마다 Send()를 사용해 병렬 작업 노드로 보냄
    return [
        Send("write_section", SectionState(
            section=s,
            transcribed_notes_file=state.transcribed_notes_file,
            urls=state.urls,
            completed_sections=[]  # 완료된 섹션 리스트를 초기화
        )) 
        for s in state.sections 
        if s.main_body
    ]

def gather_completed_sections(state: BlogState):
    """ 작성이 완료된 본문 섹션들을 모으는 함수 """

    # 1) 이미 완료된 섹션들 가져오기
    completed_sections = state.completed_sections

    # 2) 완료된 섹션 목록을 문자열로 포맷팅 (서론/결론 작성 시 참고하기 위함)
    completed_report_sections = format_sections(completed_sections)

    # 3) 포맷팅 결과를 반환
    return {"blog_main_body_sections": completed_report_sections}

def initiate_final_section_writing(state: BlogState):
    """ 서론, 결론 등 추가 조사가 필요 없는 섹션 작성을 병렬로 시작하는 함수 """

    # 1) 본문 섹션이 아닌 섹션마다 Send()를 사용해 병렬 작업 노드로 보냄
    return [
        Send("write_final_sections", SectionState(
            section=s,
            blog_main_body_sections=state.blog_main_body_sections,
            urls=state.urls,
            completed_sections=[]  # 완료된 섹션 리스트를 초기화
        )) 
        for s in state.sections 
        if not s.main_body
    ]

def compile_final_blog(state: BlogState):
    """ 모든 섹션을 합쳐 최종 블로그 글을 완성하는 함수 """

    # 1) 전체 섹션과 이미 작성이 끝난 섹션 내용 가져오기
    sections = state.sections
    completed_sections = {s.name: s.content for s in state.completed_sections}

    # 2) 섹션 순서는 그대로 유지하면서, 각 섹션에 작성된 내용을 반영
    for section in sections:
        section.content = completed_sections[section.name]

    # 3) 모든 섹션의 내용을 합쳐 최종 블로그 문자열 생성
    all_sections = "\n\n".join([s.content for s in sections])

    # 4) 최종 블로그 글 반환
    return {"final_blog": all_sections}

# ------------------------------------------------------------
# 상태 머신(Graph) 구성
builder = StateGraph(
    BlogState, 
    input=BlogStateInput, 
    output=BlogStateOutput, 
    config_schema=configuration.Configuration
)

# 1) 블로그 섹션 기획 노드 추가
builder.add_node("generate_blog_plan", generate_blog_plan)
# 2) 본문 섹션 작성 노드 추가
builder.add_node("write_section", write_section)
# 3) 최종 블로그 컴파일 노드 추가
builder.add_node("compile_final_blog", compile_final_blog)
# 4) 작성 완료 섹션 모으기 노드 추가
builder.add_node("gather_completed_sections", gather_completed_sections)
# 5) 서론/결론 섹션 작성 노드 추가
builder.add_node("write_final_sections", write_final_sections)

# ------------------------------------------------------------
# 노드들 간의 연결(Edges) 정의
builder.add_edge(START, "generate_blog_plan")
builder.add_conditional_edges("generate_blog_plan", initiate_section_writing, ["write_section"])
builder.add_edge("write_section", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_blog")
builder.add_edge("compile_final_blog", END)

# ------------------------------------------------------------
# 최종 Graph 객체 컴파일
graph = builder.compile()
#show_graph(graph)