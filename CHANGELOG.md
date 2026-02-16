# Changelog

이 프로젝트의 모든 주요 변경 사항을 기록합니다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를 기반으로 하며,
[Semantic Versioning](https://semver.org/lang/ko/)을 따릅니다.

---

## [v1.1.0] - 2026-02-16

### Added
- **G2P 기반 음소 정렬**: IPA→한글 변환 시 G2P(Grapheme-to-Phoneme) 기반 greedy alignment 도입 및 종성 호환성 검사 구현 (`ipa2kr.py`)
- **SPN 음절 복원**: MFA의 `spn`(spoken noise) 구간에 대해 텍스트 기반 음절/자모 후보정 로직 추가 (`_fill_spn_gaps`, `_try_greedy`)
- **백분율 정규화 TextGrid**: 음성 길이 기준 백분율 정규화된 TextGrid 저장 기능 및 음절 티어 생성
- **모듈 분리**: `transcriber.py`의 단일 파일 구조를 기능별 모듈로 분리 (`aligner.py`, `pitch.py`, `momel.py`, `plotting.py`)
- **데이터 전처리 스크립트**: JSON→TSV 변환(`parse_json_to_tsv.py`), 데이터 선별(`select_data.py`), 전처리(`preproc_data.py`, `preproc_data2.py`)
- **병렬 처리 개선**: MFA 배치 정렬 병렬 작업 수(`n_jobs`) 설정 및 환경 변수 기반 경로 설정 지원
- **Baikal 정렬기**: Baikal 기반 대안 정렬기 추가 (`aligner_baikal.py`)
- **테스트 도구**: 30파일 파이프라인 테스트(`prepare_test.py`), 결과 분석(`analyze_results.py`), Fix 비교(`compare_fix.py`)

### Fixed
- **구두점 처리**: MFA `.lab` 파일 생성 시 구두점 제거, 어절 경계 비교 시 구두점 무시, TextGrid 출력에서 구두점 제거
- **SPN 음절 순서 역전 방지**: spn 세그먼트에서 `syl_cursor` 전진을 제거하여 음절 순서가 바뀌는 문제 해결 (예: "진흙으로"→"흙으로진")
- **부분 정렬 보존**: greedy alignment 실패 시에도 성공한 음절까지의 결과를 보존
- **float 정밀도 보정**: TextGrid 인접 구간 경계 스냅으로 `ValueError` 방지
- **TSV BOM 처리**: UTF-8 BOM이 포함된 TSV 파일 읽기 개선
- **성별별 pitch 기본값**: 성별에 따른 pitch 범위 기본값 보장
- **오디오 파일 변환**: 비-WAV 포맷 오디오 자동 변환 기능 추가
- **Docker**: `setuptools<74` 다운그레이드로 `pkg_resources` 호환성 해결, conda 설치 메모리 절약

### Changed
- **README 업데이트**: `transcriber.py` CLI 인자 설명 및 예시를 현재 코드에 맞게 갱신
- **로거 개선**: 로거 최대 파일 크기 조정
- **`.gitignore` 정비**: CSV, PKL, TXT, 테스트 출력 파일 패턴 추가

---

## [v1.0.0] - 2025-03-18

### Added
- 초기 공개 릴리즈
- Momel 기반 억양 윤곽 추적 및 F0 목표점 추출
- MFA 기반 음성-텍스트 강제 정렬 (어절/음소 레벨)
- Pitch Doubling/Halving 자동 보정
- TextGrid 파일 및 JPEG 그래프 출력
- Docker 이미지 배포 (`linky1584/koina`)
- Gradio 웹 UI

---

## [v0.5.0] - 2025-01-15

### Added
- MFA 한국어 모델 통합
- 기본 IPA→한글 변환 (`ipa2kr.py`)
- TextGrid 생성 파이프라인

---

## [v0.0.1] - 2024-11-01

### Added
- 프로젝트 초기 구조 설정
- 기본 스크립트 뼈대

---

[v1.1.0]: https://github.com/YugwonWon/KOINA/compare/v1.0.0...v1.1.0
[v1.0.0]: https://github.com/YugwonWon/KOINA/compare/v0.5.0...v1.0.0
[v0.5.0]: https://github.com/YugwonWon/KOINA/compare/v0.0.1...v0.5.0
[v0.0.1]: https://github.com/YugwonWon/KOINA/releases/tag/v0.0.1
