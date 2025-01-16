import operator  # 파이썬 연산자 함수를 제공하는 모듈
from dataclasses import dataclass, field  # 데이터 클래스를 정의하고 필드를 다루기 위한 모듈
from pydantic import BaseModel, Field  # 데이터 검증/스키마 정의를 위한 Pydantic 클래스
from typing_extensions import Annotated, List  # 타입 힌팅용 확장 도구

# 하나의 섹션(블로그/리포트의 한 부분)을 나타내는 Pydantic 모델
class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    content: str = Field(
        description="The content of the section."
    )   
    main_body: bool = Field(
        description="Whether this is a main body section."
    )   

# 여러 섹션을 담는 Pydantic 모델
class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

# 블로그 작성 전체 상태를 담는 데이터 클래스
@dataclass(kw_only=True)
class BlogState:
    transcribed_notes_file: str   # 음성 인식을 통해 받아온 텍스트 파일 경로
    urls: List[str] = field(default_factory=list)  # 참고할 URL 리스트
    sections: list[Section] = field(default_factory=list)  # 계획된 섹션 목록
    completed_sections: Annotated[list, operator.add]  # Send() API 호출 시 사용할 키
    blog_main_body_sections: str = field(default=None)  # 작성된 본문 섹션들을 합친 문자열
    final_blog: str = field(default=None)  # 최종 블로그 글

# 외부 입력으로부터 필요한 정보를 받기 위한 데이터 클래스
@dataclass(kw_only=True)
class BlogStateInput:
    transcribed_notes_file: str  # 블로그 작성용 메모(음성 인식 등)
    urls: List[str] = field(default_factory=list)  # 참고할 URL 리스트

# 최종 아웃풋으로 최종 블로그 글을 전달하기 위한 데이터 클래스
@dataclass(kw_only=True)
class BlogStateOutput:
    final_blog: str = field(default=None)  # 최종 완성된 블로그 글

# 특정 섹션을 작성하기 위한 상태 정보를 담는 데이터 클래스
@dataclass(kw_only=True)
class SectionState:
    section: Section  # 실제 섹션 객체
    transcribed_notes_file: str = field(default=None)  # 음성 인식 파일 경로
    urls: List[str] = field(default_factory=list)  # 참고할 URL 리스트
    blog_main_body_sections: str = field(default=None)  # 본문 섹션들(다른 섹션) 내용을 모아놓은 문자열
    completed_sections: list[Section] = field(default_factory=list)  # 작성 완료된 섹션 리스트 (Send() API에서 사용)

# 본문 섹션 작성 이후에 최종 결과를 담는 데이터 클래스
@dataclass(kw_only=True)
class SectionOutputState:
    completed_sections: list[Section] = field(default_factory=list)  # 최종적으로 작성 완료된 섹션 리스트
