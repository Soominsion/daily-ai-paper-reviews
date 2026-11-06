# 작성 및 관리 방법

## 로컬 환경

Python 3.11 이상과 Git을 설치하고, VS Code에서 이 폴더를 연다. Windows PowerShell 터미널에서 다음을 실행한다. 가상환경 활성화 없이 실행할 수 있어 PowerShell 실행 정책을 바꿀 필요가 없다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/update_index.py
.\.venv\Scripts\python.exe scripts/update_index.py --check
```

VS Code의 `Python: Select Interpreter`에서 `.venv\Scripts\python.exe`를 선택한다. Markdown은 `Ctrl+Shift+V`로 미리 볼 수 있다. 수식·Mermaid의 최종 표시는 GitHub에서도 확인한다. macOS/Linux에서는 위 Python 경로 대신 `.venv/bin/python`을 사용한다.

외부 패키지는 YAML 파싱을 위한 PyYAML 하나다. 테스트는 Python 표준 라이브러리 `unittest`를 사용한다.

## GitHub 최초 업로드

GitHub에서 **Public**, 이름 `daily-ai-paper-reviews`로 빈 저장소를 만든다. README, .gitignore, License는 생성 화면에서 추가하지 않는다. 아래 `YOUR_USERNAME`을 자신의 GitHub 사용자명으로 바꾸고 이 폴더의 터미널에서 실행한다.

```powershell
git init -b main
git add .
git commit -m "Set up AI paper reading archive"
git remote add origin https://github.com/YOUR_USERNAME/daily-ai-paper-reviews.git
git push -u origin main
```

첫 커밋 전에 Git이 사용자 정보를 요구하면 `git config user.name "이름"`, `git config user.email "이메일"`을 설정한다. Push 인증은 Git Credential Manager의 GitHub 로그인 안내를 따른다.

GitHub의 **Actions → Update paper archive**에서 실행 결과를 확인한다. **Run workflow → main**으로 수동 실행할 수도 있다. 이 작업에는 별도 API 키나 개인 액세스 토큰이 필요 없다.

자동 커밋은 `GITHUB_TOKEN`의 `contents: write` 권한으로 수행한다. 저장소/조직 정책이 이를 제한하면 **Settings → Actions → General**의 권한 정책을 확인한다. `main`에 PR을 의무화하는 규칙이 있으면 봇의 직접 Push도 거부될 수 있다. 이 경우 로컬 인덱스를 함께 커밋하거나, 해당 저장소의 운영 규칙에 맞는 별도 PR 방식으로 워크플로를 변경한다.

## 함께 읽고 정리하는 방법

1. 논문과 학습 질문을 정하고, 먼저 전반적인 논문 구조를 설명한다. 문제의식과 핵심 아이디어, 모델·데이터·학습·평가의 전체 흐름과 절 구성을 잡는다.
2. Abstract부터 원문 문단을 하나씩 순서대로 번역한다. 각 문단의 용어·수식·그림을 연결해 이해하고 사용자의 질문을 다룬 뒤 다음 문단으로 넘어간다.
3. 읽는 동안 사용자가 질문하거나 궁금해한 내용과 그 답, 자기 말로 이해한 내용을 정리한다. 해결할 문제, 입력·출력, 모듈 연결, 단계별 학습 대상, 실험 조건과 남은 질문을 적고 최종 글에 반영한다.
4. 메모를 바탕으로 초안을 작성한다. 문장·bullet·도식·질문이 섞여 있어도 된다. 읽기는 끝났지만 설명 글을 아직 쓰지 않았다면 읽기 목록을 `read`로 둔다.
5. AI와 초안을 원논문에 대조하고, 사용자의 이해와 말투를 보존하면서 승인된 SAM 포맷으로 재구성한다. 수식·그림·실험 수치의 설명을 보완하고 `draft` 상태로 검토한다.
6. 사용자가 최종 글을 승인하면 `published`로 전환해 그림·출처와 함께 업로드하고, README와 Actions 결과를 확인한다. 번역·편집 메모와 미승인 초안은 공개하지 않는다.

각 절을 읽을 때는 먼저 이해에 집중하고, 전체 내용을 파악한 뒤 글의 흐름을 다듬는다. 공개 글에는 작성 과정 안내나 교정 메모를 넣지 않고, 관련 설명 옆에 원문 근거를 연결한다.

## 논문 한 편 추가하기

1. 작업 시작 시 `git pull --rebase origin main`으로 봇의 README 커밋까지 가져온다.
2. `reading-list.md`에서 논문을 골라 `queued`를 `reading`으로 바꾼다.
3. `templates/paper-review.md`를 `reviews/<대표 분야>/YYYY-MM-DD-paper-slug.md`로 복사한다. 여러 분야에 걸쳐도 파일은 하나만 만들고 `categories`에 여러 태그를 넣는다.
4. 제목·링크·작성일·학회·선정 등급을 채우고 한국어로 설명을 작성한다. 이미지는 `assets/images/<paper-slug>/`에 저장하고 출처를 기록한다. `status: "draft"`를 유지한다.
5. 초안을 마치면 읽기 목록을 `drafted`로 바꾼다. 수식, 실험 수치, 비교 조건, 선정 등급의 출처를 검토한 뒤 리뷰를 `status: "published"`, 읽기 목록을 `done`으로 바꾼다.
6. 테스트와 `python scripts/update_index.py`를 실행한다. VS Code에서 가상환경을 활성화하지 않았다면 위의 `.\.venv\Scripts\python.exe` 전체 경로를 사용한다.
7. `git add .`, `git commit -m "Add paper explanation: PAPER_TITLE"`, `git push`로 올린다. README 생성은 로컬에서 미리 해도 되고, `main` Push 후 Actions에 맡겨도 된다.

`draft`는 README 인덱스에서만 제외한다. Public 저장소에 커밋한 초안 파일 자체는 열람할 수 있다. 읽기 목록 상태는 수동으로 관리하며 인덱스 스크립트가 변경하지 않는다.

## 폴더와 메타데이터

대표 폴더는 `memory`, `personalization`, `recommendation`, `hci`, `agents`, `multimodal`, `evaluation`, `systems`, `post-training`이다. 필요하면 새 폴더를 추가해도 스크립트가 재귀적으로 찾는다. 빈 폴더의 `.gitkeep`는 Git 추적용이며 인덱싱하지 않는다.

| Field | 의미와 검증 규칙 |
|---|---|
| `title` | 논문 원제. 비어 있지 않은 한 줄 문자열 |
| `date` | 리뷰 작성일. `YYYY-MM-DD`; 학회 연도와 독립적 |
| `venue` | 학회·발표 연도. 워크숍이면 워크숍명도 명시 |
| `distinction` | 선택 항목. 확인한 등급만 기록하고 본문에 공식 출처를 링크. 미확인 시 `""` |
| `categories` | 비어 있지 않은 문자열 목록. 중복 태그 불가; 자유롭게 확장 가능 |
| `paper` | 원논문을 가리키는 절대 `https://` 또는 `http://` URL |
| `code` | 선택 항목. 공식 코드 URL. 없으면 `""` 또는 생략 |
| `status` | `draft` 또는 `published` |

날짜는 따옴표 사용을 권장하며 YAML 날짜 형식도 지원한다. 기타 문자열은 따옴표로 감싸고 내부의 `"`는 `\"`로 이스케이프한다. 후속 블로그 이전에 필요한 `authors`, `doi` 등 추가 필드는 유지할 수 있다. 별도 형식 검증은 위 필드에만 적용한다.

본문은 템플릿의 일곱 절을 출발점으로 삼되, 논문의 흐름에 맞게 바꾼다. 수식의 기호를 풀어 쓰고 입력부터 출력까지 한 예시를 따라간다. 논문이 보인 사실, 비유, 작성자의 해석을 구분한다. 그림과 표에는 Figure/Table 번호와 원문 출처를 연결하고, 외부 이미지의 저작권·이용 조건은 원출처를 따른다. 사용자가 승인한 SAM 글을 문체·구조의 기준으로 삼으며, 이후 글에도 적용할 구체적인 지침은 [AGENTS.md](AGENTS.md)에 기록한다.

## 인덱스 자동화

`scripts/update_index.py`는 모든 `reviews/**/*.md`의 YAML과 본문 존재 여부를 검사한다. 초안도 검사하므로 잘못된 메타데이터가 쌓이지 않는다. 중복 YAML 키, 필수값 누락, 잘못된 날짜·상태·URL·태그, 같은 논문의 중복 파일을 오류로 보고한다. 동일 제목 또는 동일 원문 URL을 중복으로 취급하고 arXiv의 abs/pdf 및 버전 차이도 정규화한다.

검증을 통과하면 `published` 리뷰를 작성일 내림차순으로 정렬한다. 같은 날짜는 제목, 경로 순으로 정렬해 매 실행 결과를 일정하게 유지한다. README의 `PAPER_ARCHIVE:START`와 `PAPER_ARCHIVE:END` 사이에서 총 개수와 표만 갱신하고 바깥 바이트는 보존한다. 제목은 논문, Review는 저장소 내 Markdown으로 연결한다. 등급이 있으면 Venue 옆에 함께 표시한다.

`--check`는 쓰지 않고 검사한다. 종료 코드는 정상 `0`, 인덱스 갱신 필요 `1`, 파일/메타데이터 오류 `2`이다. 검증 오류가 있으면 README를 쓰지 않는다. 오류 메시지에는 파일 경로를 표시하고 여러 파일의 오류를 모아서 보고한다.

워크플로는 관련 파일의 `main` Push, PR, 수동 실행으로 동작한다. PR은 읽기 권한으로 검사만 한다. `main`에서는 검사 성공 후 인덱스 갱신 작업에만 `contents: write`를 부여하고, README가 바뀔 때만 해당 파일을 커밋한다. 자동 커밋은 `GITHUB_TOKEN`으로 Push하므로 새 Push 워크플로를 유발하지 않으며 커밋 메시지에도 `[skip ci]`를 넣었다. [GitHub의 토큰 트리거 규칙](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)을 따른다.

동시 실행은 브랜치별로 조정한다. 실행 도중 새 커밋이 올라와 Push가 거부되면 강제 Push하지 않고 실패한다. 최신 `main`에서 수동 재실행하면 된다. 자동 커밋 후 다음 로컬 Push 전에 `git pull --rebase origin main`을 실행한다.

README의 자동 생성 경계 주석은 삭제하지 않는다. 설명·작성 지침은 이 문서, 논문 내용은 개별 리뷰에 둔다. 상대 링크와 YAML 메타데이터를 사용하므로 추후 GitHub Pages나 블로그에서 같은 원본을 활용할 수 있다.
