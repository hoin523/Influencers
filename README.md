# AI Influencer Factory

AI 버추얼 인플루언서를 대량 생산하고 직접 운영하는 콘텐츠 파이프라인.

## Quick Start

```bash
# 1. 의존성 설치
pip install -e ".[dev]"

# 2. .env 파일에 API 키 설정
cp .env.example .env
# ANTHROPIC_API_KEY, OPENAI_API_KEY 입력

# 3. 백엔드 실행
python main.py

# 4. 대시보드 실행 (별도 터미널)
streamlit run dashboard.py
```

## 환경

- **Python:** 3.11+ (3.12 OK)
- **로컬 GPU:** 없음 (Intel UHD 770)
- **이미지 생성:** RunPod 클라우드 GPU (A5000 24GB 권장)
- **ComfyUI:** RunPod에서 원격 실행, API로 연결
- **LLM:** Claude API + GPT-4o (폴백)

## 프로젝트 구조

```
main.py              FastAPI 백엔드
dashboard.py         Streamlit 리뷰 대시보드
models.py            DB 스키마 + 상태 머신
database.py          SQLite 엔진
config.py            환경 설정
services/
  llm.py             LLM 콘텐츠 생성
  comfyui.py         ComfyUI API 이미지 생성
  pipeline.py        파이프라인 오케스트레이션
personas/            페르소나 YAML (source of truth)
prompts/             LLM 프롬프트 템플릿
workflows/           ComfyUI 워크플로우 JSON
assets/              페르소나별 이미지 에셋
tests/               pytest 테스트 (31개)
```

## RunPod ComfyUI 설정

1. RunPod에서 GPU Pod 생성 (A5000 24GB 권장)
2. ComfyUI 템플릿 선택 또는 수동 설치
3. IP-Adapter FaceID Plus + 동반 LoRA 다운로드
4. `.env`의 `COMFYUI_URL`을 RunPod 주소로 변경
5. SSH 포트 포워딩: `ssh -L 8188:localhost:8188 root@<pod-ip>`
