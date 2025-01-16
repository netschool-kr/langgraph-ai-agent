from .state import GraphState
from .element import Element
import base64
import os
from .base import BaseNode
from langchain_core.documents import Document


IMAGE_TYPES = ["figure", "chart"]
TEXT_TYPES = ["text", "equation", "caption", "paragraph", "list", "index", "heading1"]
TABLE_TYPES = ["table"]


class CreateElementsNode(BaseNode):
    def __init__(self, verbose=False, add_newline=True, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.add_newline = add_newline
        self.newline = "\n" if add_newline else ""

    def _save_base64_image(self, base64_str, basename, page_num, element_id, directory):
        """base64 인코딩된 이미지를 파일로 저장하는 함수"""
        img_filename = f"{basename}_Page_{page_num}_Index_{element_id}.png"
        # 원본 파일과 동일한 디렉토리에 저장
        img_filepath = os.path.join(directory, img_filename)
        img_data = base64.b64decode(base64_str)
        with open(img_filepath, "wb") as f:
            f.write(img_data)
        return img_filepath

    def execute(self, state: GraphState) -> GraphState:
        post_processed_elements = []
        directory = os.path.dirname(state["filepath"])
        base_filename = os.path.splitext(os.path.basename(state["filepath"]))[0]

        for element in state["elements_from_parser"]:
            elem = None
            if element["category"] in ["footnote", "header", "footer"]:
                continue

            if element["category"] in ["equation"]:
                # (markdown only)
                # equation
                elem = Element(
                    category=element["category"],
                    content=element["content"]["markdown"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    page=element["page"],
                    id=element["id"],
                )

            elif element["category"] in ["table"]:
                # (markdown + image crop/image save)
                # table
                image_filename = self._save_base64_image(
                    element["base64_encoding"],
                    base_filename,
                    element["page"],
                    element["id"],
                    directory,
                )
                elem = Element(
                    category=element["category"],
                    content=element["content"]["markdown"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    base64_encoding=element["base64_encoding"],
                    image_filename=image_filename,
                    page=element["page"],
                    id=element["id"],
                    coordinates=element["coordinates"],
                )

            elif element["category"] in ["figure", "chart"]:
                # (markdown + image crop/image save)
                # figure, chart
                image_filename = self._save_base64_image(
                    element["base64_encoding"],
                    base_filename,
                    element["page"],
                    element["id"],
                    directory,
                )

                elem = Element(
                    category=element["category"],
                    content=element["content"]["markdown"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    base64_encoding=element["base64_encoding"],
                    image_filename=image_filename,
                    page=element["page"],
                    id=element["id"],
                    coordinates=element["coordinates"],
                )
            elif element["category"] in ["heading1"]:
                # (text w/ heading)
                # heading1
                elem = Element(
                    category=element["category"],
                    content=f'# {element["content"]["text"]}{self.newline}',
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    page=element["page"],
                    id=element["id"],
                )
            elif element["category"] in ["caption", "paragraph", "list", "index"]:
                # (text)
                # caption, paragraph, list
                elem = Element(
                    category=element["category"],
                    content=element["content"]["text"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    page=element["page"],
                    id=element["id"],
                )

            if elem is not None:
                post_processed_elements.append(elem)

        return GraphState({"elements": post_processed_elements})

# extracted_image_entities는 이전 단계에서 이미지 요소에 대해 엔티티 추출 작업을 수행한 결과물을 담고 있는 리스트입니다. 
# ImageEntityExtractorNode 같은 노드를 거치면, 원래 elements 리스트 속 특정 이미지 요소들(figure, chart, table 등)에 
# 대응하는 새로운 Element 객체들이 extracted_image_entities 리스트에 생성되며, 이들 객체는 entity 필드에 추출된 엔티티 
# 정보가 포함되어 있습니다.

# MergeEntityNode는 이 extracted_image_entities 리스트를 참조하여 원래 elements 리스트에서 동일한 id를 가진 요소를 찾아 
# entity 필드를 업데이트하는 역할을 합니다. 다시 말해, extracted_image_entities는 엔티티 추출 결과를 임시로 모아둔 상태이고, 
# MergeEntityNode가 이를 원래 요소들과 합쳐 최종적으로 모든 엔티티 정보가 elements에 반영되도록 하는 것입니다.


class MergeEntityNode(BaseNode):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def execute(self, state: GraphState) -> GraphState:
        elements = state["elements"]

        for elem in state["extracted_image_entities"]:
            for e in elements:
                if elem.id == e.id:
                    e.entity = elem.entity
                    break

        for elem in state["extracted_table_entities"]:
            for e in elements:
                if elem.id == e.id:
                    e.entity = elem.entity
                    break

        return GraphState({"elements": elements})

# 이 코드는 ReconstructElementsNode라는 노드를 정의하고 있습니다. 이 노드는 이전 단계에서 만들어진 elements 리스트와 
# 페이지별 텍스트 정보(texts_by_page), 그리고 이미지 및 테이블 정보 등을 종합하여 reconstructed_elements라는 구조로 다시 
# 정리하는 역할을 합니다. 정리된 결과는 페이지별로 텍스트, 이미지, 테이블 정보를 재구성한 딕셔너리 형태로 반환합니다.
class ReconstructElementsNode(BaseNode):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def _add_src_to_markdown(self, image_filename):
        """마크다운 이미지 문법에 src 경로를 추가하는 함수"""
        abs_image_path = os.path.abspath(image_filename)
        image_md = f"![](file:///{abs_image_path})"
        return image_md

    def execute(self, state: GraphState) -> GraphState:
        elements = state["elements"]
        filepath = state["filepath"]

        pages = sorted(list(state["texts_by_page"].keys()))
        max_page = pages[-1]

        reconstructed_elements = dict()
        for page_num in range(max_page + 1):
            reconstructed_elements[int(page_num)] = {
                "text": "",
                "image": [],
                "table": [],
            }

        for elem in elements:
            if elem.category in TABLE_TYPES:
                table_elem = {
                    "content": elem.content + "\n\n" + elem.entity,
                    "metadata": {
                        "table": elem.content,
                        "entity": elem.entity,
                        "page": elem.page,
                        "source": filepath,
                    },
                }
                reconstructed_elements[elem.page]["table"].append(table_elem)
            elif elem.category in IMAGE_TYPES:
                image_elem = {
                    "content": self._add_src_to_markdown(elem.image_filename)
                    + "\n\n"
                    + elem.entity,
                    "metadata": {
                        "image": self._add_src_to_markdown(elem.image_filename),
                        "entity": elem.entity,
                        "page": elem.page,
                        "source": filepath,
                    },
                }
                reconstructed_elements[elem.page]["image"].append(image_elem)
            elif elem.category in TEXT_TYPES:
                reconstructed_elements[elem.page]["text"] += elem.content

        return GraphState({"reconstructed_elements": reconstructed_elements})

#  이 코드는 LangChainDocumentNode라는 클래스로, 이전 단계에서 재구성된 요소(reconstructed_elements)를 
# LangChain에서 사용하는 Document 객체 리스트로 변환하는 노드입니다. 이렇게 하면 추후 LangChain 기반의 
# LLM 처리(예: 검색, 요약, 번역)에 사용할 수 있습니다

class LangChainDocumentNode(BaseNode):
    def __init__(self, splitter, verbose=False):
        super().__init__(verbose)
        self.splitter = splitter

    def execute(self, state: GraphState) -> GraphState:
        reconstructed_elements = state["reconstructed_elements"]
        filepath = state["filepath"]
        documents = []
        for page_num, page_data in reconstructed_elements.items():
            text = page_data["text"]
            split_texts = self.splitter.split_text(text)
            for split_text in split_texts:
                documents.append(
                    Document(
                        page_content=split_text,
                        metadata={"page": page_num, "source": filepath},
                    )
                )
            images = page_data["image"]
            for image in images:
                documents.append(
                    Document(page_content=image["content"], metadata=image["metadata"])
                )
            tables = page_data["table"]
            for table in tables:
                documents.append(
                    Document(page_content=table["content"], metadata=table["metadata"])
                )
                documents.append(
                    Document(
                        page_content=table["metadata"]["entity"],
                        metadata=table["metadata"],
                    )
                )

        return GraphState({"documents": documents})
