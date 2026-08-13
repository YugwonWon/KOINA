

# KOINA (Korean Intonation Annotator)

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e4d95ca-9c29-46cb-b543-d0d3bf245f25" width="400">
</p>

KOINA(Korean Intonation Annotator, 한국어 억양 주석기)는 한국어 억양을 자동으로 주석해주는 도구입니다. 음성 데이터를 빠르고 일관된 방식으로 처리하여 연구자의 수작업 부담을 줄이고, 전사의 객관성과 일관성을 확보하며, 대규모 자연 발화 음성 자료를 효율적으로 처리할 수 있게 해줍니다.
이 프로젝트는 한국어 음성학 및 억양 연구 분야에서 활용될 목적으로 설계되었으며, Momel 알고리즘 기반 음높이(F0) 윤곽 추출, Pitch Doubling/Halving 오류 수정, TextGrid 파일과 JPEG 그래프 출력 등 다양한 기능을 제공합니다.
개발 과정은 원유권(2025)의 연구 결과를 토대로 이루어졌으며, Docker 및 Python 환경에서 손쉽게 사용할 수 있도록 구성하였습니다.

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

   - 음높이(F0) 목표점 추출: 알고리즘을 통해 F0 윤곽을 추출
   - 삼차 연결 곡선(spline) 적용: 목표점을 기반으로 곡선 형태의 억양 윤곽을 재현
   - TCoG 추출: 음높이 무게 중심
2. **음성-텍스트 강제 정렬**

   - 어절 및 음소 레벨 정렬
   - 정렬 결과를 TextGrid 파일로 생성하여, 구간별 분석이 용이
3. **F0 목표점 최소화**

   - 기울기 기반 단순화: F0 목표점을 제거하여 효율적인 윤곽 생성
   - 지각적 일관성 유지: 목표점을 단순화하여도 청각적 인식은 원본과 동일 수준 유지
4. **Pitch Doubling/Halving 보정**

   - 자동 감지 및 수정: 기존 F0 윤곽에서 배증 혹은 반감으로 튀는 오차를 보정하고 안정화
5. **시각화 및 결과 저장**

   - TextGrid 파일 출력: Praat 등 음성 분석 툴과 호환 가능한 형식
   - JPEG 그래프 저장: 분석 결과를 바로 확인 가능한 이미지로 제공
   - 음성 합성 WAV 저장: 음성 합성된 WAV 파일 저장

---

## 2. Docker 환경에서의 사용 방법

1. KOINA는 **공식 Docker 이미지**를 통해 별도 의존성 설치 없이 바로 실행할 수 있습니다. 로컬 PC·서버 어디서든 동일한 절차로 웹 UI([http://localhost:40080](http://localhost:40080))에 접근할 수 있습니다.

| 단계                                                       | 터미널 명령어 및 설명                                                                                                                                                                                                                                                           |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Docker Desktop / Docker Engine 설치**           | [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)                                                                                                                                                                                                         |
| **2. 이미지 받기**                                   | ``docker pull linky1584/koina:latest``                                                                                                                                                                                                                                          |
| **3-A. Intel/AMD PC에서 실행**                       | ``docker run --rm -p 40080:40080 -v /your/audio/path:/koina/out --name koina linky1584/koina:latest``                                                                                                                                                                           |
| **3-B. Apple Silicon(M1/M2) 맥 / ARM 서버에서 실행** | QEMU 에뮬레이션으로 amd64 이미지를 구동합니다.<br /><br />`docker run --rm --platform linux/amd64 -p 40080:40080 -v /your/audio/path:/koina/out --name koina linky1584/koina:latest`<br /><br />**Tip:** 첫 구동 시 다운로드·초기화 때문에 2-3분 정도 걸릴 수 있습니다. |
| **4. 웹 UI 접속**                                    | 브라우저에 [http://localhost:40080](http://localhost:40080) 입력 → 파일 업로드·실행                                                                                                                                                                                               |
| **5. 컨테이너 종료**                                 | 터미널에서 `Ctrl + C` 또는 다른 쉘에서 `docker stop koina`                                                                                                                                                                                                                  |

2. 정상 접속 화면

<p align="center">
  <img src="https://github.com/user-attachments/assets/20dce17c-3b1f-40cb-954c-e8a26bf68dc2" width="1237" height="643" alt="Image">
</p>

3-1. **csv(tsv)파일 입력**

- 아래는 CSV/TSV 파일 예시입니다. 각 열(column)의 이름은 반드시 `filename`, `sex`, `text` 를 포함해야, CSV의 경우 쉼표로 구분하고, TSV의 경우 탭으로 구분하는 것만 다릅니다.

**TSV 예시**

```tsv
   filename    sex    text
   SDRW2200000001.1.1.1.wav    M    어 여기서 학교 얘기가 나와서
```

3-2. 음성-텍스트 정렬 코퍼스 폴더 구조 및 화자 설정

KOINA는 `도커가 실행될 때 유저가 볼륨 마운트(-v 명령어)한 폴더가 out/` 디렉토리에 연결되고, 이 폴더를 기준으로 WAV 파일을 읽어들입니다. 그리고 MFA 배치 정렬 시 **폴더별 화자**를 인식해 병렬 처리(job)를 자동 분배합니다. 하지만 별도 화자 구분을 가정하지 않고 주제, 대화별 폴더를 구성하여도 상관 없습니다.

- 단일 폴더에 모든 WAV 배치

```text
  out/
    a.wav
    b.wav
    c.wav
```

- 화자별 하위 폴더 분리 예시

```text
out/
  speaker1/
    a.wav
    b.wav
  speaker2/
    c.wav
    d.wav
  speaker3/
    e.wav
```

---

## 3. Python 로컬 환경에서 직접 실행하기 (Linux, Conda 권장)

아래에서는 `./src/transcribe/transcriber.py` 스크립트를 중심으로, Python 환경에서 KOINA 억양 전사 모듈을 실행하는 방법을 소개합니다.

1. **사전 준비**

   - 먼저 현재 리포지토리를 클론(clone)합니다. 예시로, 로컬 머신에서 다음 명령어를 사용할 수 있습니다:

   ```bash
   git clone https://github.com/YugwonWon/KOINA.git
   ```
2. Miniforge/Anaconda 설치 (한 번만)

   - 설치 가이드는 https://docs.conda.io/projects/miniconda/en/latest/ 참조
   - Windows PowerShell·macOS Terminal·Linux bash 모두 동일합니다.
3. Conda 가상환경 생성 & 패키지 설치

   ```bash
   # 3-1) 새 환경 생성
   conda create -y -n koina python=3.10
   conda activate koina

   # 3-2) 채널 설정
   conda config --add channels conda-forge
   conda config --set channel_priority strict

   # 3-3) 주요 패키지 + MFA 3.2.1 설치
   mamba install -y \
     montreal-forced-aligner=3.2.1 \
     numpy pandas scipy matplotlib tqdm \
     textgrid pydub ffmpeg-python

   # 3-4) 한국어 토크나이저 및 부가 패키지
   pip install python-mecab-ko jamo \
               gradio soundfile==0.12.1 praat-parselmouth

   # 3-5) MFA 한국어 모델 다운로드
   mfa model download dictionary korean_mfa
   mfa model download acoustic   korean_mfa
   ```
4. 스크립트 실행

- transcriber.py에는 억양 전사를 위한 핵심 로직이 포함되어 있으며, 터미널(또는 커맨드라인)에서 argument를 지정해 실행할 수 있습니다. 예시 명령어는 다음과 같습니다:

```bash
python ./src/transcribe/transcriber.py \
  data/input.tsv \
  --wav_root_dir data/source-audio \
  --save_dir out/results \
  --n_jobs 4
```

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `tsv_file` (위치 인자) | 입력 TSV 파일 경로 (`filename`, `sex`, `text` 컬럼 포함) | `data/133_parsed_output_sample.tsv` |
| `--wav_root_dir` | WAV 파일이 있는 디렉토리 경로 | `data/source-audio` |
| `--save_dir` | 출력 TextGrid 파일들이 저장될 디렉토리 경로 | `out/results` |
| `--n_jobs` | 병렬 처리할 프로세스 수 | `4` |

---

## 4. 이슈 보고 및 피드백

프로젝트는 아직 완성 단계가 아니며, 다양한 실제 음성 분석 현장에서 얻은 피드백을 통해 개선될 여지가 많습니다.

- **버그 리포트**: 예기치 않은 오류가 발생했을 경우, 재현 가능한 단계와 로그 정보, 사용한 입력 파일(일부 샘플)을 보내주시면 문제 파악에 큰 도움이 됩니다.
- **기능 요청**: 추가로 필요한 기능이나 개선 사항이 있다면 자유롭게 이슈(이야기)를 남겨주세요.

---

## 5. Reference

- 원유권(2025). 한국어 억양 자동 주석기 개발 연구, 건국대학교 대학원 박사학위논문. <a href="https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=1cbfcc7b60e1739bffe0bdc3ef48d419&keyword=%ED%95%9C%EA%B5%AD%EC%96%B4%20%EC%96%B5%EC%96%91%20%EC%9E%90%EB%8F%99%20%EC%A3%BC%EC%84%9D%EA%B8%B0%20%EA%B0%9C%EB%B0%9C%20%EC%97%B0%EA%B5%AC" target="_blank">[링크]</a>

## 6. Cites

- 아래의 DOI 또는 BibTeX를 통해 인용해주세요.
- DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18666913.svg)](https://doi.org/10.5281/zenodo.18666913)
- BibTeX:

```bibtex
@misc{KOINA,
  author = {Won, YuGwon},
  title = {KOINA: Korean Intonation Annotator},
  year = {2026},
  version = {v1.1.0},
  doi = {\url{https://doi.org/10.5281/zenodo.18666913}},
  journal = {Github repository},
  note = {\url{https://github.com/YugwonWon/KOINA}}
}
```

---

🐳 Found it useful? Sponsor us or buy me a coffee!

<p align="left">
   <a href="https://github.com/sponsors/YugwonWon" target="_blank">
    <img 
      src="https://img.shields.io/github/sponsors/YugwonWon?style=for-the-badge" 
      alt="GitHub Sponsors" />
  </a>
  <a href="https://buymeacoffee.com/yugwonwon" target="_blank">
    <img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-yellow?logo=buy-me-a-coffee&style=for-the-badge"
         alt="Buy Me A Coffee" />
  </a>
</p>
