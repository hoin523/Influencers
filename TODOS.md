# TODOS

## TODO-001: 수동 포스팅 헬퍼 스크립트
**Priority:** Medium
**Status:** Planned
**What:** 대시보드에서 승인된 콘텐츠의 이미지 파일 열기 + 캡션/해시태그 클립보드 복사를 자동화하는 간단한 스크립트.
**Why:** 수동 포스팅이 인플루언서 3개 × 플랫폼 3개 = 하루 30-45분. "혼자 다수 운영" 가치가 이 병목에 제한됨. API 없이도 포스팅 시간 50%+ 단축 가능.
**Context:** Outside Voice가 지적. 현재 MVP는 수동 포스팅이지만, 인플루언서 수 증가 시 스케일링 한계. pyperclip + webbrowser 모듈로 간단 구현 가능.
**Depends on:** Week 3 대시보드 완료 후
**Added:** 2026-04-03 by /plan-eng-review

## TODO-002: 플랫폼별 이미지 후처리
**Priority:** Medium
**Status:** Planned
**What:** 생성된 이미지를 플랫폼 요구사항에 맞게 리사이즈/크롭 (Instagram 1080x1350, TikTok 1080x1920 등) + EXIF 메타데이터 처리.
**Why:** ComfyUI 생성 이미지를 그대로 올릴 수 없음. 해상도/비율이 플랫폼마다 다르고, EXIF 메타데이터에 AI 생성 흔적이 남을 수 있음.
**Context:** Outside Voice가 지적. Pillow 라이브러리로 구현. 파이프라인의 ComfyUI → 저장 단계에 후처리 레이어 추가. 플랫폼별 프리셋 YAML로 관리.
**Depends on:** ComfyUI 이미지 생성 완료 후 (Week 2+)
**Added:** 2026-04-03 by /plan-eng-review

## TODO-003: GPU 스펙 문서화
**Priority:** High (Week 1 시작 전)
**Status:** Done
**What:** 사용 중인 GPU 모델, VRAM, CUDA 버전, 드라이버 버전을 README 또는 설정 문서에 기록.
**Resolution:** 로컬에 NVIDIA GPU 없음 (Intel UHD 770 내장 그래픽만). RunPod 클라우드 GPU 사용 결정.
**Added:** 2026-04-03 by /plan-eng-review
