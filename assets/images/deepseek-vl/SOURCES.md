# DeepSeek-VL architecture diagram

`architecture.svg`와 `architecture.png`는 이 리뷰를 위해 직접 만든 구조 개념도다. 원논문의 그림이나 외부 이미지 파일을 복제하지 않았다. 동일한 종류의 개념에는 동일한 도형을 쓴다. 사다리꼴은 모듈·연산, 평행사변형은 데이터·feature·token을 나타낸다. 배치 차원과 일부 정규화 연산은 생략했다.

- Hybrid encoder의 전체 원리: [DeepSeek-VL §3.1](https://arxiv.org/html/2403.05525v2#S3.SS1)
- SAM-B image encoder, bilinear interpolation, convolutional downsampling: [공식 sam.py](https://github.com/deepseek-ai/DeepSeek-VL/blob/main/deepseek_vl/models/sam.py)
- 두 해상도 입력과 feature 반환: [공식 clip_encoder.py](https://github.com/deepseek-ai/DeepSeek-VL/blob/main/deepseek_vl/models/clip_encoder.py)
- 경로별 Linear → feature concatenation → GeLU → shared Linear: [공식 projector.py](https://github.com/deepseek-ai/DeepSeek-VL/blob/main/deepseek_vl/models/projector.py)
- 공개 7B의 `input_dim=1024`, `n_embed=4096`, `depth=2`, `low_high_hybrid_split_mlp_gelu`: [공식 config.json](https://huggingface.co/deepseek-ai/deepseek-vl-7b-base/blob/main/config.json)

그림은 공개 7B checkpoint의 실제 adaptor 차원을 따른다. 각 경로가 1024 → 2048 Linear를 거친 뒤 feature 차원으로 합쳐져 576 × 4096이 되고, shared Linear도 4096 → 4096이다. 논문에서 언급한 576 × 2048은 두 원시 1024차원 feature를 결합한 설명으로 구분한다.

SAM 원형의 prompt encoder와 mask decoder는 DeepSeek-VL의 이 경로에 사용하지 않는다. 1.3B 공개 모델은 SigLIP 단일 encoder를 사용한다.

외부 그림·사진·아이콘·폰트 파일을 포함하지 않는다. SVG 텍스트는 시스템 폰트로 표시한다. 참고한 공식 구현의 코드는 [MIT License](https://github.com/deepseek-ai/DeepSeek-VL/blob/main/LICENSE-CODE), 모델 weights는 별도의 [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-VL/blob/main/LICENSE-MODEL)를 따른다.
