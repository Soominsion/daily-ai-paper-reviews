# Qwen-VL 구조 이미지

`architecture.png`는 이 리뷰를 위해 직접 그린 구조 개념도다. 논문의 그림이나 외부 이미지 파일을 복제하지 않았다.

- 구조: [Qwen-VL 원논문 §2.1–2.2](https://arxiv.org/html/2308.12966v3#S2.SS1)
- 패치 크기 14, 인코더 너비 1664, LLM 입력·시각 출력 차원 4096, 입력 해상도 448: [Qwen 공식 모델 설정](https://huggingface.co/Qwen/Qwen-VL/blob/main/config.json)
- 256개 query, 이미지 특징 projection 및 cross-attention: [Qwen 공식 배포 코드의 Resampler와 VisionTransformer](https://huggingface.co/Qwen/Qwen-VL/blob/da34fee90e0c97711c8933e0039f2b0f30b9b9fd/visual.py)

원은 파라미터, 사다리꼴은 처리 모듈, 평행사변형은 입력·특징·토큰·출력이다. 점선은 파라미터의 사용 관계, 실선은 데이터 흐름이다. 정규화, 특수 토큰, multi-head 내부 구성 등의 세부 연산은 생략했다.

- 작성일: 2026-11-05 (Asia/Seoul).
- SHA-256: `9e793ef0c9c37683d86db43ecd525e896cb2754a153b99859a8d074f8c486056`.
