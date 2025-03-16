# KOINA (Korean Intonation Annotator)
![ ](./docs/images/logo.jpeg)

KOINA(Korean Intonation Annotator, 한국어 억양 주석기)는 한국어 억양을 자동으로 주석해주는 도구입니다. 음성 데이터를 빠르고 일관된 방식으로 처리하여 연구자의 수작업 부담을 덜어주기 위한 도구입니다.
이 프로젝트는 한국어 음성학 및 억양 연구 분야에서 활용될 목적으로 설계되었으며, Momel 알고리즘 기반 음높이(F0) 윤곽 추출, Pitch Doubling/Halving 오류 수정, TextGrid 파일과 JPEG 그래프 출력 등 다양한 기능을 제공합니다.
개발 과정은 원유권(2025)의 연구 결과를 토대로 이루어졌으며, Docker 및 Python 환경에서 손쉽게 활용할 수 있도록 구성하였습니다.

---

## 목차
1. **주요 기능**
2. **Docker 환경에서의 사용 방법**
3. **Python 환경에서의 사용 방법**
4. **이슈 보고 및 피드백**
6. **reference**
7. **cites**

---

## 1. 주요 기능
1. **Momel 기반 억양 윤곽 추적**
   - **음높이(F0) 목표점 추출**: 알고리즘을 통해 음높이 윤곽을 검출
   - **삼차 연결 곡선(spline) 적용**: 목표점을 기반으로 자연스럽게 억양 궤적을 재현
   - TCoG 추출: 음높이 무게 중심

2. **음성-텍스트 강제 정렬**
   - 어절 및 음소 레벨 정렬
   - 정렬 결과를 TextGrid 파일로 생성하여, 구간별 분석이 용이

3. **음높이 목표점 최소화**
   - **기울기 기반 단순화**: 과도한 목표점을 제거하여 효율적인 윤곽 생성
   - **지각적 일관성 유지**: 단순화에도 불구, 청각적 인식은 원본과 동일 수준 유지

4. **Pitch Doubling/Halving 보정**
   - 자동 감지 및 수정: 기존 F0 궤적에서 2배 혹은 절반으로 튀는 오차를 검출하고 안정화

5. **시각화 및 결과 저장**
   - **TextGrid 파일 출력**: Praat 등 음성 분석 툴과 호환 가능한 형식
   - **JPEG 그래프 저장**: 분석 결과를 바로 확인 가능한 이미지로 제공

---

## 2. Docker 환경에서의 사용 방법
KOINA는 공식 Docker 이미지를 통해 간편하게 실행할 수 있습니다. 로컬 환경에서 별도로 구축해야 할 의존성 없이, Docker 컨테이너 실행만으로 웹 인터페이스에 접근할 수 있습니다.

1. **도커 설치**
   - 로컬 머신에 Docker를 먼저 설치합니다. (운영체제별 설치 방법은 [Docker Docs](https://docs.docker.com/get-docker/) 참고)

2. **이미지 받기 (pull)**
   ```bash
   docker pull linky1584/koina:latest

3. **도커 실행**
   ```
   docker run --rm -p 7861:7861 -v /your/audio/path:/koina/out --name koina linky1584/koina:latest