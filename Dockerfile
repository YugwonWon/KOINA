# 1. 기본 이미지 설정 (Ubuntu 22.04, 64bit)
FROM ubuntu:22.04

# 2. 환경 변수 설정
ENV LC_ALL=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 3. 패키지 리스트 업데이트 & 필수 패키지 설치
RUN dpkg --add-architecture i386 \
    && apt-get update -y \
    && apt-get install -y \
       tzdata \
       python3.10 \
       python3-pip \
       libsndfile1:i386 \
       libavcodec-extra:i386 \
       ffmpeg \
       curl \
       git \
       build-essential \
       supervisor \
       fonts-nanum \
       coreutils \
       net-tools

# 4. 필요한 파이썬 패키지 개별 설치 (requirements.txt 없이)
RUN pip3 install --upgrade pip && pip3 install \
    gradio \
    numpy \
    pandas \
    scipy \
    matplotlib \
    protobuf==3.20 \
    praat-parselmouth \
    soundfile==0.12.1 \
    pydub==0.25.1 \
    ffmpeg-python \
    grpcio \
    grpcio-reflection \
    textgrid \
    tqdm

# 5. 작업 디렉토리 설정 및 코드 복사
WORKDIR /koina
COPY src/analyze /koina/src/analyze
COPY src/client /koina/src/client
COPY src/lib /koina/src/lib
COPY src/transcribe /koina/src/transcribe
COPY src/utils /koina/src/utils
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
ENV PYTHONPATH="src:src/lib:src/client:src/analyze:src/transcribe"

# 6. 포트 설정 (Gradio UI)
EXPOSE 7861

# 7. 로그 디렉토리 생성 (필수)
RUN mkdir -p /koina/logs

# 8. Supervisord 실행 (front.py 자동 실행)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
