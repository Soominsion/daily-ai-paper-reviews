# LLaVA 구조 개념도

`architecture.svg`는 초기 LLaVA의 구조를 설명하기 위해 직접 작성한 벡터 개념도다. 원논문 Figure 1이나 외부 이미지 파일을 복제하지 않았다.

- 구조·표현 연결: [Visual Instruction Tuning §4.1, Figure 1, 식 (1)](https://arxiv.org/html/2304.08485v2#S4.SS1)
- 단계별 학습 대상: [원논문 §4.2](https://arxiv.org/html/2304.08485v2#S4.SS2)
- 선형 projection과 시각 인코더 고정: [2023-04-25 시점의 공식 학습 코드](https://github.com/haotian-liu/LLaVA/blob/a6b01a246683e82cd1af9c6c0bd7338995cef1c5/llava/train/train.py)

`N`은 시각 토큰 수, `d_v`는 시각 특징 차원, `d_L`은 언어 모델 입력 차원이다. 행렬의 각 행을 토큰 하나로 표현했다. 도형의 색은 흐름을 구분하기 위한 것이며, 학습 여부는 아래의 단계별 설명을 따른다. 텍스트 임베딩 층도 LLM 파라미터에 속한다. 이미지 전처리·특수 토큰·대화 이력·출력 확률 계산은 생략했다.

외부 그림·사진·아이콘·폰트 파일을 포함하지 않는다. SVG의 텍스트는 시스템 폰트로 표시한다.

작성일: 2026-11-07 (Asia/Seoul).
