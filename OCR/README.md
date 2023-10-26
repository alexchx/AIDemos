# OCR

Based on RapidOCR and can be used for simple scenarios, for complex scenarios like table recognition, please use PaddleOCR instead.

## Setup
Add config file .env with proper configurations.
```text
OPENAI_API_BASE=xxxxxx
OPENAI_API_KEY=xxxxxx
```

## Run

```commandline
pip install -r requirements.txt
python main.py
```

## References
- https://github.com/yuanjie-ai/ChatLLM#chatocr
- https://github.com/RapidAI/RapidOCR
- https://github.com/PaddlePaddle/PaddleOCR
- https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/docs/quickstart.md
- https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/README_ch.md
- https://github.com/Layout-Parser/layout-parser
