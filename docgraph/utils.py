
from .state import GraphState
from .base import BaseNode
import pymupdf
import os
from PIL import Image
import matplotlib.pyplot as plt
import os
from typing_extensions import Annotated, TypedDict
import json
import pickle
import matplotlib

# GUI 없는 환경에서는 Agg 백엔드 사용
#matplotlib.use('Agg')
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


import xml.etree.ElementTree as ET


def load_xml_files(directory, encoding="utf-8"):
    """
    주어진 디렉토리에서 .xml 파일을 로드하고, XML 내용을 파싱하여 반환합니다.

    Args:
        directory (str): XML 파일이 포함된 디렉토리 경로.
        encoding (str): XML 파일을 디코딩할 때 사용할 인코딩. 기본값은 'utf-8'.

    Returns:
        list: 파싱된 XML 내용의 리스트. 각 항목은 딕셔너리 형태로 파일 이름과 내용을 포함합니다.
    """
    h1_docs = []
    for file in os.listdir(directory):
        if file.endswith(".xml"):  # XML 파일만 처리
            file_path = os.path.join(directory, file)
            try:
                with open(file_path, "rb") as f:  # 바이너리 모드로 파일 읽기
                    content = f.read()
                    root = ET.fromstring(content)  # XML 파싱
                    text = ET.tostring(root, encoding="unicode")  # 유니코드로 변환
                    h1_docs.append({"file": file, "content": text})
            except Exception as e:
                print(f"Error parsing file {file_path}: {e}")
    return h1_docs

class SplitPDFFilesNode(BaseNode):

    def __init__(self, batch_size=10, test_page=None, **kwargs):
        super().__init__(**kwargs)
        self.name = "SplitPDFNode"
        self.batch_size = batch_size
        self.test_page = test_page

    def execute(self, state: GraphState) -> GraphState:
        """
        입력 PDF를 여러 개의 작은 PDF 파일로 분할합니다.

        :param state: GraphState 객체, PDF 파일 경로와 배치 크기 정보를 포함
        :return: 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체
        """
        # PDF 파일 경로와 배치 크기 추출
        filepath = state["filepath"]

        # PDF 파일 열기
        input_pdf = pymupdf.open(filepath)
        num_pages = len(input_pdf)
        self.log(f"파일의 전체 페이지 수: {num_pages} Pages.")

        if self.test_page is not None:
            if self.test_page < num_pages:
                num_pages = self.test_page

        ret = []
        # PDF 분할 작업 시작
        for start_page in range(0, num_pages, self.batch_size):
            # 배치의 마지막 페이지 계산 (전체 페이지 수를 초과하지 않도록)
            end_page = min(start_page + self.batch_size, num_pages) - 1

            # 분할된 PDF 파일명 생성
            input_file_basename = os.path.splitext(filepath)[0]
            output_file = f"{input_file_basename}_{start_page:04d}_{end_page:04d}.pdf"
            self.log(f"분할 PDF 생성: {output_file}")

            # 새로운 PDF 파일 생성 및 페이지 삽입
            with pymupdf.open() as output_pdf:
                output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
                output_pdf.save(output_file)
                ret.append(output_file)

        # 원본 PDF 파일 닫기
        input_pdf.close()

        # 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체 반환
        return GraphState(split_filepaths=ret)

# LangChain의 핵심 컴포넌트들을 임포트합니다
from langchain_core.prompts import PromptTemplate
from pydantic import Field, BaseModel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import get_model_name, LLMs
import re
import markdown
# GPT-4 모델을 사용하기 위한 모델명을 가져옵니다


# 번역된 텍스트를 담기 위한 Pydantic 모델 클래스 정의
class TranslatedText(BaseModel):
    # 번역된 텍스트를 저장할 필드 정의
    translated_text: str = Field(description="The translated text of the given text")

class TranslateNode(BaseNode):

    def __init__(self, model_name = get_model_name(LLMs.GPT4), verbose=False, language="Korean", **kwargs):
        super().__init__(**kwargs)
        self.name = "TranslateNode"
        super().__init__(verbose=verbose, **kwargs)
        self.model_name = model_name
        self.language="Korean"
        # 번역된 텍스트를 저장할 필드 정의
        # Pydantic 모델을 사용하는 출력 파서 생성
        output_parser = PydanticOutputParser(pydantic_object=TranslatedText)
        # 번역을 위한 프롬프트 템플릿 정의
        self.prompt = PromptTemplate.from_template(
            """You are a translation expert. Translate the <given_text> into Korean.
        [IMPORTANT] Keep the <given_text>'s markdown format.

        ###

        <given_text>
        {text}
        </given_text>"""
        )
        # ChatGPT 모델 인스턴스 생성 및 구조화된 출력 설정
        self.llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(TranslatedText)

        # 프롬프트와 LLM을 연결하는 체인 생성
        self.chain = self.prompt | self.llm
    """
    번역 모듈을 추가하는 함수입니다.
    주어진 상태(state)의 텍스트 요소들을 한국어로 번역합니다.

    Args:
        state (GraphState): 파싱된 상태 객체

    Returns:
        dict: 번역된 요소들이 포함된 상태 딕셔너리
    """


    def execute(self, state: GraphState):
        # 번역이 필요한 요소들을 담을 리스트 생성
        translated_elements = []
        # 상태에서 번역이 필요한 카테고리의 요소들만 선택
        for element in state["elements_from_parser"]:
            # 번역이 필요한 카테고리들을 지정
            if element["category"] in [
                "paragraph",
                "index",
                "heading1",
                "header",
                "footer",
                "caption",
                "list",
                "footnote",
            ]:
                translated_elements.append(element)

        # 배치 크기 설정 (한 번에 처리할 요소 수)
        BATCH_SIZE = 50
        # 번역 결과를 저장할 리스트
        all_translated_results = []
        # 배치 단위로 번역 처리
        for i in range(0, len(translated_elements), BATCH_SIZE):
            # 현재 배치의 요소들 추출
            batch = translated_elements[i : i + BATCH_SIZE]
            # 배치 데이터 준비
            batch_data = [{"text": text["content"]["markdown"]} for text in batch]
            # 재시도 횟수 설정
            trial = 3
            while trial > 0:
                try:
                    # 배치 번역 실행
                    batch_results = self.chain.batch(batch_data)
                    break
                except Exception as e:
                    print(e)
                    trial -= 1
                    continue
            # 번역 결과 출력
            for result in batch_results:
                print(result)
            # 전체 결과 리스트에 현재 배치 결과 추가
            all_translated_results.extend(batch_results)

        # 번역된 텍스트를 원본 요소에 업데이트
        for i, result in enumerate(all_translated_results):
            translated_elements[i]["content"]["markdown"] = result.translated_text
            if self.language == "Korean":
                new_html = markdown.markdown(result.translated_text)
                new_html = new_html.replace("&lt;given_text&gt;", "").replace("&lt;/given_text&gt;", "")
                new_html = new_html.replace("<code>","").replace("</code>","").replace("<pre>","").replace("</pre>","")
                translated_elements[i]["content"]["html"] = new_html


        # 업데이트된 상태 반환
        return GraphState(elements_from_parser= state["elements_from_parser"], language="Korean")

def save_graph(save_file_name, graph):
    """
    현재 그래프 상태를 파일에 저장합니다.
    Args:
        save_file_name (str): 그래프 상태를 저장할 파일 경로
        graph (StateGraph): 저장할 그래프 객체
    """
    try:
        with open(save_file_name, "wb") as f:
            pickle.dump(graph, f)
        print(f"✅ 그래프 상태 저장 완료: {save_file_name}")
    except Exception as e:
        print(f"❌ 그래프 상태 저장 실패: {e}")

def load_graph(save_file_name):
    """
    저장된 그래프 상태를 파일에서 복원합니다.
    Args:
        save_file_name (str): 그래프 상태가 저장된 파일 경로
    Returns:
        StateGraph: 복원된 그래프 객체
    """
    if not os.path.exists(save_file_name):
        print(f"❌ 그래프 상태 파일을 찾을 수 없습니다: {save_file_name}")
        return None

    try:
        with open(save_file_name, "rb") as f:
            graph = pickle.load(f)
        print(f"🔄 그래프 상태 복원 완료: {save_file_name}")
        return graph
    except Exception as e:
        print(f"❌ 그래프 상태 복원 실패: {e}")
        return None

    
    
def save_state(state_file, metadata, chunk_msg):
    """
    현재 상태를 JSON 파일에 저장합니다.
    Args:
        metadata (dict): 그래프 실행 중의 메타데이터
        chunk_msg (str): 현재 처리 중인 메시지
    """
    state = {
        "node": metadata["langgraph_node"],
        "chunk_msg": chunk_msg.content,
    }
    with open(state_file, "w") as f:
        json.dump(state, f)
    print(f"✅ 상태 저장 완료: {state}")

def load_state(state_file):
    """
    저장된 상태를 JSON 파일에서 복원합니다.
    Returns:
        dict: 저장된 상태 정보
    """
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
        print(f"🔄 상태 복원 완료: {state}")
        return state
    return None