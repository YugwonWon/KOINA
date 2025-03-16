# KOINA (Korean Intonation Annotator)
<p align="center">
  <img src="https://github.com/user-attachments/assets/6e4d95ca-9c29-46cb-b543-d0d3bf245f25" width="400">
</p>

KOINA(Korean Intonation Annotator, 한국어 억양 주석기)는 한국어 억양을 자동으로 주석해주는 도구입니다. 음성 데이터를 빠르고 일관된 방식으로 처리하여 연구자의 수작업 부담을 줄이고, 전사의 객관성과 일관성을 확보하며, 대규모 자연 발화 음성 자료를 효율적으로 처리할 수 있게 해줍니다.
이 프로젝트는 한국어 음성학 및 억양 연구 분야에서 활용될 목적으로 설계되었으며, Momel 알고리즘 기반 음높이(F0) 윤곽 추출, Pitch Doubling/Halving 오류 수정, TextGrid 파일과 JPEG 그래프 출력 등 다양한 기능을 제공합니다.
개발 과정은 원유권(2025)의 연구 결과를 토대로 이루어졌으며, Docker 및 Python 환경에서 손쉽게 활용할 수 있도록 구성하였습니다.

(아직 개발중인 단계로 기능이 추가되거나 수정될 수 있습니다.)
---

## 목차
1. **주요 기능**
2. **Docker 환경에서의 사용 방법**
3. **Python 환경에서의 사용 방법**
4. **이슈 보고 및 피드백**
5. **reference**
6. **cites**

---

## 1. 주요 기능
1. **Momel 기반 억양 윤곽 추적**
   - 음높이(F0) 목표점 추출: 알고리즘을 통해 음높이 윤곽을 검출
   - 삼차 연결 곡선(spline) 적용: 목표점을 기반으로 자연스럽게 억양 궤적을 재현
   - TCoG 추출: 음높이 무게 중심

2. **음성-텍스트 강제 정렬**
   - 어절 및 음소 레벨 정렬
   - 정렬 결과를 TextGrid 파일로 생성하여, 구간별 분석이 용이

3. **음높이 목표점 최소화**
   - 기울기 기반 단순화: 과도한 목표점을 제거하여 효율적인 윤곽 생성
   - 지각적 일관성 유지: 단순화에도 불구, 청각적 인식은 원본과 동일 수준 유지

4. **Pitch Doubling/Halving 보정**
   - 자동 감지 및 수정: 기존 F0 궤적에서 배증 혹은 반감으로 튀는 오차를 검출하고 안정화

5. **시각화 및 결과 저장**
   - TextGrid 파일 출력: Praat 등 음성 분석 툴과 호환 가능한 형식
   - JPEG 그래프 저장: 분석 결과를 바로 확인 가능한 이미지로 제공

---

## 2. Docker 환경에서의 사용 방법
KOINA는 공식 Docker 이미지를 통해 간편하게 실행할 수 있습니다. 로컬 환경에서 별도로 구축해야 할 의존성 없이, Docker 컨테이너 실행만으로 웹 인터페이스에 접근할 수 있습니다.

1. **도커 설치**
   - 로컬 머신에 Docker를 먼저 설치합니다. (운영체제별 설치 방법은 [Docker Docs](https://docs.docker.com/get-docker/) 참고)

2. **이미지 받기 (pull)**
   ```bash
   docker pull linky1584/koina:latest

3. **도커 실행**
   - 아래 명령어(CMD)는 사용자가 로컬에 있는 오디오 파일들이 위치한 폴더 경로를 `/your/audio/path` 부분에 넣어 실행하는 예시입니다.
   ```
   docker run --rm -p 7861:7861 -v /your/audio/path:/koina/out --name koina linky1584/koina:latest

4. **웹 인터페이스 실행**
   - 도커를 실행했다면, 인터넷 창을 열고 아래 주소를 입력해서 클라이언트를 실행합니다.
   ```
   http://localhost:7861
   ```

   <p align="center">
     <img src="https://github.com/user-attachments/assets/5887a465-1946-46f5-9ed8-2d0251ac2d16" width="1397">
   </p>

5. **csv(tsv)파일 입력**
   - 아래는 CSV/TSV 파일 예시입니다. 각 열(컬럼)은 반드시 `wav_filename`, `sex`, `text` 순서를 지켜야 하며, CSV의 경우 쉼표로 구분하고, TSV의 경우 탭으로 구분하는 것만 다릅니다.

   #### TSV 예시
   ```tsv
   wav_filename    sex    text
   SDRW2200000001.1.1.1.wav    M    어 여기서 학교 얘기가 나와서

## 3. Python 환경에서의 사용 방법

아래에서는 `./src/transcribe/transcriber.py` 스크립트를 중심으로, Python 환경에서 KOINA 억양 전사 모듈을 실행하는 방법을 소개합니다.

---

1. **사전 준비**
   -  먼저 현재 리포지토리를 클론(clone)합니다. 예시로, 로컬 머신에서 다음 명령어를 사용할 수 있습니다:

   ```bash
   git clone https://github.com/YugwonWon/KOINA.git
   ```

2. **가상환경(venv) 생성 및 라이브러리 설치**
   - 연구 환경이나 서버별로 파이썬 라이브러리가 상이할 수 있으므로, 가능한 한 독립적인 가상환경을 마련하는 편이 좋습니다. 다음과 같은 절차로 venv를 생성하고 필요한 라이브러리를 설치할 수 있습니다:
   ```bash
   # 가상환경(venv) 생성
   python -m venv venv

   # 가상환경 활성화 (Windows, Mac/Linux 명령어 상이)
   # 예: Mac/Linux
   source venv/bin/activate

   # 예: Windows
   .\venv\Scripts\activate

   # requirements 설치
   pip install -r requirements.txt
   ```

3. **스크립트 실행**
   - transcriber.py에는 억양 전사를 위한 핵심 로직이 포함되어 있으며, 터미널(또는 커맨드라인)에서 아규먼트를 지정해 실행할 수 있습니다. 예시 명령어는 다음과 같습니다:
   ```bash
   python ./src/transcribe/transcriber.py \
    data/input.tsv \
    out \
    --momel_path src/lib/momel/momel_linux
   ```

## 4. 이슈 보고 및 피드백
프로젝트는 아직 완성 단계가 아니며, 다양한 실제 음성 분석 현장에서 얻은 피드백을 통해 개선될 여지가 많습니다.  
- **버그 리포트**: 예기치 않은 오류가 발생했을 경우, 재현 가능한 단계와 로그 정보, 사용한 입력 파일(일부 샘플)을 보내주시면 문제 파악에 큰 도움이 됩니다.  
- **기능 요청**: 추가로 필요한 기능이나 개선 사항이 있다면 자유롭게 이슈(이야기)를 남겨주세요.  

---

## 5. Reference
본 프로젝트의 개발 배경과 알고리즘 상세는 원유권(2025)의 연구 결과를 참조하였으며, Docker, Python, Praat, Ubuntu 등의 여러 오픈소스 툴이 활용되었습니다.

- 원유권(2025). 한국어 억양 자동주석기 개발 연구, 건국대학교 일반대학원 박사학위논문. [링크]
- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth:
A Python interface to Praat. Journal of Phonetics, 71, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001
- Paul Boersma & David Weenink (2025): Praat: doing phonetics by
computer [Computer program]. Version 6.4.23, retrieved 27
October 2024 from https://www.praat.org.
- Helmke, M., Graner, A., Rankin, K., Hill, B. M., & Bacon, J. (2013). The
Official Ubuntu Book (7th ed.) . Upper Saddle River, NJ: Prentice
Hall.

---

## 6. Cites
 - 아래의 DOI 또는 BibTeX를 통해 인용해주세요.
 - DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15034862.svg)](https://doi.org/10.5281/zenodo.15034862)
 - BibTeX:
```bibtex
@misc{KOINA,
  author = {Won, YuGwon},
  title = {KOINA: Korean Intonation Annotator},
  year = {2025},
  version = {v1.0.0},
  doi = {\url{https://doi.org/10.5281/zenodo.15034862}},
  journal = {Github repository},
  note = {\url{https://github.com/YugwonWon/KOINA}}
}
