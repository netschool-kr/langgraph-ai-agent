import os  # 운영체제와 상호작용하기 위한 라이브러리
from dataclasses import dataclass, fields  # 데이터 클래스를 정의하고 필드를 다루기 위한 라이브러리
from typing import Any, Optional  # 타입 힌팅을 위한 도구들

from langchain_core.runnables import RunnableConfig  # langchain_core에서 RunnableConfig 가져오기
from dataclasses import dataclass  # dataclass 모듈(위에서 이미 임포트했지만, 중복 선언 무해)

# 블로그 기본 구조를 정의한 문자열 상수
DEFAULT_BLOG_STRUCTURE = """The blog post should follow this strict three-part structure:

1. Introduction (max 1 section)
   - Start with ### Key Links and include user-provided links  
   - Brief overview of the problem statement
   - Brief overview of the solution/main topic
   - Maximum 100 words

2. Main Body (exactly 2-3 sections)
    - Each section must:
      * Cover a distinct aspect of the main topic
      * Include at least one relevant code snippet
      * Be 150-200 words
    - No overlap between sections

3. Conclusion (max 1 section)
   - Brief summary of key points
   - Key Links
   - Clear call to action
   - Maximum 150 words"""

@dataclass(kw_only=True)
class Configuration:
    """
    챗봇에서 설정할 수 있는 필드들을 정의한 데이터 클래스.
    blog_structure 필드에 기본값으로 DEFAULT_BLOG_STRUCTURE 문자열을 사용한다.
    """
    blog_structure: str = DEFAULT_BLOG_STRUCTURE
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """
        주어진 RunnableConfig 객체로부터 Configuration 인스턴스를 생성하는 클래스 메서드.
        config가 존재하고 그 안에 'configurable'이라는 키가 있으면 해당 딕셔너리를 사용,
        그렇지 않다면 빈 딕셔너리를 사용한다.
        """
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        # dataclass로 선언된 모든 필드를 순회하면서,
        # (1) 환경 변수(os.environ)에 필드 이름과 동일한 대문자 키가 있는지 확인하고,
        # (2) 없으면 configurable 딕셔너리에 필드 이름과 일치하는 키가 있는지 확인한다.
        # 두 곳에서 가져온 값 중 유효한 것을 values 딕셔너리에 저장한다.
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        # values 중 값이 None이 아닌 항목만 걸러서, Configuration 인스턴스를 생성해 반환한다.
        return cls(**{k: v for k, v in values.items() if v})
