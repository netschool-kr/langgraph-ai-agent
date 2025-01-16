

블로그 글을 깔끔하게 작성하는 일은 전통적으로 많은 시간과 노력을 요구합니다. 좋은 아이디어가 떠올라도, 이를 체계적인 글로 완성하기까지 큰 간극이 존재하죠. Robo Blogger는 이 간극을 줄이기 위해 아이디어 수집과 글 구조화를 분리하여, 더 쉽고 빠르게 완성도 높은 글을 작성하도록 돕습니다.

1. 설치 및 실행 준비
robo_blogger.zip 다운로드

robo_blogger.zip을 받아 원하는 폴더(예: C:/AI/)에 압축을 풀어주세요.
압축을 풀면 robo_blogger/ 디렉터리가 생성됩니다.
Visual Studio Code에서 열기

VSCode를 실행한 뒤, ‘열기(Open Folder)’ 메뉴를 사용해 robo_blogger/ 폴더를 선택합니다.
Python 가상환경 설정(권장)

VSCode 내 터미널 혹은 독립된 터미널에서 다음을 입력해 가상환경을 생성하고 활성화합니다:

python -m venv venv
source venv/bin/activate
(Windows 환경일 경우, venv\Scripts\activate 명령어로 활성화)
필요 패키지 설치

robo_blogger 폴더 안에 있는 requirements.txt(또는 pyproject.toml)가 있다면, 다음 명령어로 의존성을 설치합니다:
pip install -r requirements.txt
또는, uvx를 사용하는 방법이 README에 안내되어 있다면 해당 과정을 참고해도 됩니다.
API 키 설정(Anthropic, OpenAI 등)

.env.example 파일을 .env로 복사하고, 필요한 API 키를 기입합니다.
Anthropic Claude를 사용할 경우, ANTHROPIC_API_KEY를 .env에 넣어두거나 아래 예시처럼 환경 변수로 설정합니다:

export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
(Windows에서는 set ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY)
2. Robo Blogger 작동 방식
Robo Blogger가 해결하려는 문제는 간단합니다:

아이디어 수집은 음성으로 간편히 하고,
글 작성은 자동으로 구조화·세분화하여 적절히 분할된 섹션으로 완성.
Voice Capture (음성 캡처)

Flowvoice 같은 앱을 이용해 녹음을 텍스트로 옮긴 후, notes 폴더에 텍스트 파일(예: audio_dictation.txt)을 저장합니다.
Planning (기획)

Claude 3.5 Sonnet(Anthropic)을 통해 방금 만든 텍스트를 토대로 전체 글의 개요(섹션 구성)를 생성합니다.
Writing (작성)

사전에 정의된 블로그 구조나 URL 자료를 참고해, 자동으로 글을 작성해 줍니다.
기존 Report mAIstro 프로젝트 아이디어를 기반으로 하되 블로그 작성에 최적화되었습니다.

3. blog_run.py 실행하기
프로젝트 폴더(robo_blogger/)에는 blog_run.py라는 스크립트가 있을 것입니다. 이 스크립트를 실행하면 다음 과정이 진행됩니다:

입력 데이터 설정
음성 인식 파일 이름(예: agent.txt)과 참고할 URL 목록, 블로그 구조 템플릿(선택)을 지정합니다.
LangGraph 상태 머신(StateGraph) 호출
agent/graph.py 안에 정의된 흐름대로 섹션 기획, 본문 작성, 서론·결론 작성, 최종 합치기 과정을 순차 또는 병렬로 진행합니다.
최종 블로그 글 출력
작성이 완료된 블로그 포스트 전체를 콘솔에 출력하거나, 변수로 반환받을 수 있습니다.

4. Quickstart 요약
robo_blogger.zip 다운로드 후 압축 해제
VSCode에서 robo_blogger 폴더 열기
(선택) 가상환경 설정 후 pip install -r requirements.txt
.env 파일에 API 키(예: ANTHROPIC_API_KEY) 설정
notes/ 폴더에 음성 인식 파일(예: audio_dictation.txt, agent.txt) 배치
blog_run.py 실행:

콘솔 출력 확인: 최종 블로그 글이 표시됨

5. 추가 옵션
Optional Inputs

문서 URL: 글 작성 시 참고할 웹페이지(예: 기술 문서 링크)를 전달 가능
블로그 구조 템플릿: 글을 특정 형식으로 작성하도록 지정 (예: 제품 업데이트, 칼럼 형태 등)
Customization 예시

Report mAIstro처럼, 문서나 URL을 통해 내용 정확도를 높일 수 있습니다.
여러 유형의 블로그 템플릿(예: ‘Product Update’, ‘Perspective’)을 기획하여 재활용할 수 있습니다.

6. 마무리
robo_blogger는 음성 아이디어를 자동으로 구조화된 블로그 글로 빠르게 전환해주며,
URL, 블로그 템플릿 등 추가 자료를 통해 보다 신뢰도 높고 풍부한 콘텐츠를 생성할 수 있습니다.
VSCode에서 blog_run.py를 실행해보면서, 실제로 음성 메모 → 프로페셔널한 글 제작 과정을 체험해보세요.
