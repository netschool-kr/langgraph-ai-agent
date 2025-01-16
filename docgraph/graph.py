import os
import sys
# 현재 디렉토리를 시스템 경로에 추가
sys.path.insert(0, os.path.abspath("."))
from typing import Any, Dict, List, Callable
from docgraph.utils import SplitPDFFilesNode, TranslateNode
from docgraph.state import GraphState
from docgraph.upstage import (
    DocumentParseNode,
    PostDocumentParseNode,
    WorkingQueueNode,
    continue_parse,
)
from langgraph.graph import StateGraph,START, END
from langgraph.checkpoint.memory import MemorySaver

from docgraph.preprocessing import (
    CreateElementsNode,
    MergeEntityNode,
    ReconstructElementsNode,
    LangChainDocumentNode,
)
from docgraph.export import ExportHTML, ExportMarkdown, ExportTableCSV, ExportImage
from docgraph.extractor import (
    PageElementsExtractorNode,
    ImageEntityExtractorNode,
    TableEntityExtractorNode,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph


def create_parse_graph(
    batch_size: int = 30,
    test_page: int = None,
    verbose: bool = True,
):
    split_pdf_node = SplitPDFFilesNode(
        batch_size=batch_size, test_page=test_page, verbose=verbose
    )

    document_parse_node = DocumentParseNode(
        api_key=os.environ["UPSTAGE_API_KEY"], verbose=verbose
    )

    post_document_parse_node = PostDocumentParseNode(verbose=verbose)
    working_queue_node = WorkingQueueNode(verbose=verbose)

    # LangGraph 생성
    workflow = StateGraph(GraphState)

    # 노드들을 정의합니다.
    workflow.add_node("split_pdf_node", split_pdf_node)
    workflow.add_node("document_parse_node", document_parse_node)
    workflow.add_node("post_document_parse_node", post_document_parse_node)
    workflow.add_node("working_queue_node", working_queue_node)

    # 각 노드들을 연결합니다.
    workflow.add_edge("split_pdf_node", "working_queue_node")
    workflow.add_conditional_edges(
        "working_queue_node",
        continue_parse,
        {True: "document_parse_node", False: "post_document_parse_node"},
    )
    workflow.add_edge("document_parse_node", "working_queue_node")

    workflow.set_entry_point("split_pdf_node")

    document_parse_graph = workflow.compile(checkpointer=MemorySaver())
    return document_parse_graph

# 이 그래프는 다양한 형식으로 문서를 내보내는(export) 워크플로우를 정의한다.
def create_export_graph(
    ignore_new_line_in_text=True,
    show_image_in_markdown=False,
    verbose=True
):
    # GraphState를 다루는 StateGraph 인스턴스를 생성
    export_graph = StateGraph(GraphState)

    # 이미지, HTML, Markdown, CSV로 문서를 내보내는 노드들을 생성
    # verbose 옵션을 통해 진행 상황을 출력할지 결정할 수 있다.
    export_image = ExportImage(verbose=verbose)
    export_html = ExportHTML(
        ignore_new_line_in_text=ignore_new_line_in_text, verbose=verbose
    )
    export_markdown = ExportMarkdown(
        ignore_new_line_in_text=ignore_new_line_in_text,
        show_image=show_image_in_markdown,
        verbose=verbose,
    )
    export_table_csv = ExportTableCSV(verbose=verbose)


    # 그래프에 노드 등록
    # 각 노드는 문서를 특정 형식으로 변환하는 역할을 수행한다.
    export_graph.add_node("export_image", export_image)
    export_graph.add_node("export_html", export_html)
    export_graph.add_node("export_markdown", export_markdown)
    export_graph.add_node("export_table_to_csv", export_table_csv)

    # 노드 간 엣지 연결
    # export_image 노드가 실행된 후, 결과를 HTML, Markdown, CSV 내보내기 노드로 전달한다.
    export_graph.add_edge("export_image", "export_html")
    export_graph.add_edge("export_html", "export_markdown")
    export_graph.add_edge("export_markdown", "export_table_to_csv")
    export_graph.add_edge("export_table_to_csv", END)

    export_graph.set_entry_point("export_image")

    # 그래프의 시작 노드를 export_image로 설정
    # 즉, 이미지를 내보내는 단계부터 워크플로우를 시작한다.
    c_graph = export_graph.compile(checkpointer=MemorySaver())
    return c_graph

from langchain_teddynote.models import get_model_name, LLMs

def create_translate_graph(verbose=True, model_name=get_model_name(LLMs.GPT4), language="Korean"):
    # GraphState를 다루는 StateGraph 인스턴스를 생성
    translate_graph = StateGraph(GraphState)
    translate = TranslateNode(model_name, language="Korean", verbose=verbose)


    # 그래프에 노드 등록
    # 각 노드는 문서를 특정 형식으로 변환하는 역할을 수행한다.
    translate_graph.add_node("translate", translate)

    # HTML, Markdown, CSV 내보내기 노드는 최종적으로 그래프를 종료(END)한다.
    translate_graph.add_edge("translate", END)

    translate_graph.set_entry_point("translate")

    # 그래프의 시작 노드를 export_image로 설정
    # 즉, 이미지를 내보내는 단계부터 워크플로우를 시작한다.
    c_graph = translate_graph.compile(checkpointer=MemorySaver())
    return c_graph


def create_document_process_graph(batch_size=30, test_page=None, verbose=True):
    # PDF를 분할하는 노드 생성
    # batch_size=30: 한번에 처리할 페이지 수, test_page=None: 모든 페이지 처리, verbose=True: 처리 상황 출력
    split_pdf_node = SplitPDFFilesNode(batch_size=batch_size, test_page=test_page, verbose=verbose)
    
    # Upstage API를 통해 문서를 파싱하는 노드
    # 환경 변수로부터 UPSTAGE_API_KEY를 읽어와 인증 후 파싱 진행
    document_parse_node = DocumentParseNode(
        api_key=os.environ["UPSTAGE_API_KEY"], verbose=verbose
    )
    # 문서 파싱 이후 후처리를 수행하는 노드
    post_document_parse_node = PostDocumentParseNode(verbose=verbose)
    # 작업 큐를 관리하는 노드로, 남은 페이지 처리 여부를 판단하고 다음 단계로 라우팅
    working_queue_node = WorkingQueueNode(verbose=verbose)

    # 첫 번째 워크플로우 생성
    graph = StateGraph(GraphState)
    
     # 노드 등록
    graph.add_node("split_pdf_node", split_pdf_node)
    graph.add_node("document_parse_node", document_parse_node)
    graph.add_node("post_document_parse_node", post_document_parse_node)
    graph.add_node("working_queue_node", working_queue_node)

    # 노드 간 흐름 정의
    # split_pdf_node → working_queue_node: 분할된 PDF 페이지 목록을 큐로 전달
    graph.add_edge("split_pdf_node", "working_queue_node")
    # working_queue_node에서 continue_parse 함수를 통해 남은 페이지 처리 여부 판단
    # True: 아직 처리할 페이지 있음 → document_parse_node로 이동
    # False: 처리할 페이지 없음 → post_document_parse_node로 이동(파싱 후 마무리 단계)    
    graph.add_conditional_edges(
        "working_queue_node",
        continue_parse,
        {True: "document_parse_node", False: "post_document_parse_node"},
    )
    # document_parse_node 처리 후 다시 working_queue_node로 돌아가 다음 페이지 처리 여부 판단    
    graph.add_edge("document_parse_node", "working_queue_node")
    # 그래프 시작점 설정: split_pdf_node에서 워크플로우 시작
    graph.set_entry_point("split_pdf_node")
    parser_graph = graph.compile()

    # 후처리 단계에 필요한 노드 생성
    # CreateElementsNode: 파싱 결과로부터 요소(텍스트, 이미지, 표 등)를 추출/생성
    create_elements_node = CreateElementsNode(verbose=verbose)
    # Export 계열 노드: 파싱 결과물을 다양한 형식(HTML, Markdown, CSV)으로 내보내는 노드
    export_html = ExportHTML(verbose=verbose)
    export_markdown = ExportMarkdown(verbose=verbose)
    export_table_csv = ExportTableCSV(verbose=verbose)
    
    # PageElementsExtractorNode: 각 페이지에서 추출한 요소들을 기반으로 구조적 정보 추출
    page_elements_extractor_node = PageElementsExtractorNode(verbose=verbose)
    # ImageEntityExtractorNode: 이미지 관련 엔티티를 추출하는 노드
    image_entity_extractor_node = ImageEntityExtractorNode(verbose=verbose)
    # TableEntityExtractorNode: 테이블 관련 엔티티를 추출하는 노드
    table_entity_extractor_node = TableEntityExtractorNode(verbose=verbose)
    
    # MergeEntityNode: 이미지나 표 등 추출된 엔티티들을 병합하여 하나의 구조화된 데이터로 정리
    merge_entity_node = MergeEntityNode(verbose=True)
    # ReconstructElementsNode: 병합된 엔티티를 기반으로 원본 문서 구조를 재구성
    reconstruct_elements_node = ReconstructElementsNode(verbose=True)
    # LangChainDocumentNode: 재구성된 문서 요소를 LangChain Document 형식으로 변환하고,
    # RecursiveCharacterTextSplitter를 사용해 긴 문서를 chunking    
    langchain_document_node = LangChainDocumentNode(
        verbose=True,
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
    )

    # 후처리 워크플로우 생성
    post_process_graph = StateGraph(GraphState)

    # 후처리 단계에 필요한 노드 등록
    # document_parse 노드에 앞서 컴파일한 parser_graph를 사용
    post_process_graph.add_node("document_parse", parser_graph)
    post_process_graph.add_node("create_elements_node", create_elements_node)
    post_process_graph.add_node("export_html", export_html)
    post_process_graph.add_node("export_markdown", export_markdown)
    post_process_graph.add_node("export_table_csv", export_table_csv)
    post_process_graph.add_node(
        "page_elements_extractor_node", page_elements_extractor_node
    )
    post_process_graph.add_node(
        "image_entity_extractor_node", image_entity_extractor_node
    )
    post_process_graph.add_node(
        "table_entity_extractor_node", table_entity_extractor_node
    )
    post_process_graph.add_node("merge_entity_node", merge_entity_node)
    post_process_graph.add_node(
        "reconstruct_elements_node", reconstruct_elements_node
    )
    post_process_graph.add_node("langchain_document_node", langchain_document_node)

    # 엣지 연결 (후처리 단계의 흐름 정의)
    # document_parse → create_elements_node: 파싱이 완료된 문서를 요소화
    post_process_graph.add_edge("document_parse", "create_elements_node")
    # create_elements_node에서 생성된 요소를 기반으로 HTML, Markdown, CSV 내보내기 가능
    post_process_graph.add_edge("create_elements_node", "export_html")
    post_process_graph.add_edge("create_elements_node", "export_markdown")
    post_process_graph.add_edge("create_elements_node", "export_table_csv")
    # create_elements_node 결과를 페이지 단위로 요소 추출
    post_process_graph.add_edge(
        "create_elements_node", "page_elements_extractor_node"
    )
    # 페이지 요소에서 이미지와 테이블 추출 분기
    post_process_graph.add_edge(
        "page_elements_extractor_node", "image_entity_extractor_node"
    )
    post_process_graph.add_edge(
        "page_elements_extractor_node", "table_entity_extractor_node"
    )
    post_process_graph.add_edge("image_entity_extractor_node", "merge_entity_node")
    # HTML, Markdown, CSV 내보내기 완료 시 워크플로우 종료
    post_process_graph.add_edge("export_html", END)
    post_process_graph.add_edge("export_markdown", END)
    post_process_graph.add_edge("export_table_csv", END)
    # 테이블 엔티티 추출 후 엔티티 병합 단계로 이동
    post_process_graph.add_edge("table_entity_extractor_node", "merge_entity_node")
    # 엔티티 병합 완료 후 재구성 노드로 이동
    post_process_graph.add_edge("merge_entity_node", "reconstruct_elements_node")
    # 재구성된 요소를 LangChain Document로 변환 후 종료
    post_process_graph.add_edge(
        "reconstruct_elements_node", "langchain_document_node"
    )
    post_process_graph.add_edge("langchain_document_node", END)

     # 후처리 그래프의 시작 지점을 문서 파싱 완료 단계로 설정
    post_process_graph.set_entry_point("document_parse")

     # 체크포인트 관리를 위한 MemorySaver 인스턴스 생성
    memory = MemorySaver()
    # 후처리 그래프 컴파일 및 반환
    c_graph = post_process_graph.compile(checkpointer=memory)
    return c_graph

def stream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    LangGraph의 실행 결과를 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig): 실행 설정
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """
    prev_node = ""
    for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
        curr_node = metadata["langgraph_node"]

        # node_names가 비어있거나 현재 노드가 node_names에 있는 경우에만 처리
        if not node_names or curr_node in node_names:
            # 콜백 함수가 있는 경우 실행
            if callback:
                callback({"node": curr_node, "content": chunk_msg.content})
            # 콜백이 없는 경우 기본 출력
            else:
                # 노드가 변경된 경우에만 구분선 출력
                if curr_node != prev_node:
                    print("\n" + "=" * 50)
                    print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                    print("- " * 25)
                print(chunk_msg.content, end="", flush=True)

            prev_node = curr_node

def format_namespace(namespace):
    return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

def invoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    LangGraph 앱의 실행 결과를 예쁘게 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig): 실행 설정
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """


    # subgraphs=True 를 통해 서브그래프의 출력도 포함
    for namespace, chunk in graph.stream(
        inputs, config, stream_mode="updates", subgraphs=True
    ):
        for node_name, node_chunk in chunk.items():
            # node_names가 비어있지 않은 경우에만 필터링
            if len(node_names) > 0 and node_name not in node_names:
                continue

            # 콜백 함수가 있는 경우 실행
            if callback is not None:
                callback({"node": node_name, "content": node_chunk})
            # 콜백이 없는 경우 기본 출력
            else:
                print("\n" + "=" * 50)
                formatted_namespace = format_namespace(namespace)
                if formatted_namespace == "root graph":
                    print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                else:
                    print(
                        f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                    )
                print("- " * 25)

                # 노드의 청크 데이터 출력
                for k, v in node_chunk.items():
                    if isinstance(v, BaseMessage):
                        v.pretty_print()
                    elif isinstance(v, list):
                        for list_item in v:
                            if isinstance(list_item, BaseMessage):
                                list_item.pretty_print()
                            else:
                                print(list_item)
                    elif isinstance(v, dict):
                        for node_chunk_key, node_chunk_value in node_chunk.items():
                            print(f"{node_chunk_key}:\n{node_chunk_value}")
                print("=" * 50)
