########################
# 1. Base Image
########################
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    CONDA_DIR=/opt/conda

########################
# 2. APT packages
########################
# ca-certificates 를 먼저 깔고 update-ca-certificates 실행
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg2 && \
    update-ca-certificates && \
    apt-get install -y --no-install-recommends \
        tzdata ffmpeg libsndfile1 fontconfig fonts-nanum && \
    rm -rf /var/lib/apt/lists/*

########################
# 3. Miniconda 설치
########################
ARG MF_VER=25.3.0-3
ARG MF_URL=https://github.com/conda-forge/miniforge/releases/download/${MF_VER}/Miniforge3-${MF_VER}-Linux-x86_64.sh
RUN curl -fsSL "$MF_URL" -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR" && \
    rm /tmp/miniforge.sh && \
    "$CONDA_DIR/bin/conda" clean -afy


########################
# 4. MFA 전용 env 생성
########################
# Set default channels to conda-forge to avoid main/r ToS issues
# Also, add CONDA_PKGS_DIRS to a writable location
RUN ${CONDA_DIR}/bin/conda config --set auto_activate_base false && \
    ${CONDA_DIR}/bin/conda config --add channels conda-forge && \
    ${CONDA_DIR}/bin/conda config --set channel_priority strict && \
    ${CONDA_DIR}/bin/conda create -y -n mfa python=3.10 && \
    ${CONDA_DIR}/bin/conda run -n mfa pip install --upgrade pip

# 4-1. MFA + 필수 라이브러리 설치 (conda/pip 혼합)
RUN ${CONDA_DIR}/bin/conda run -n mfa conda install -y \
        montreal-forced-aligner==3.2.1 \
        numpy pandas scipy matplotlib tqdm \
        textgrid pydub ffmpeg-python && \
    ${CONDA_DIR}/bin/conda run -n mfa pip install joblib==1.1.0 && \
    ${CONDA_DIR}/bin/conda run -n mfa pip install gradio soundfile==0.12.1 praat-parselmouth

# 4-2. 한국어 모델 받기
RUN ${CONDA_DIR}/bin/conda run -n mfa mfa model download dictionary korean_mfa && \
    ${CONDA_DIR}/bin/conda run -n mfa mfa model download acoustic korean_mfa

########################
# 6. 코드 복사
########################
WORKDIR /koina
COPY src/ ./src/
ENV PYTHONPATH=/koina/src

########################
# 7. 포트 & 로그
########################
EXPOSE 40080

ENV PATH=/opt/conda/envs/mfa/bin:/opt/conda/bin:$PATH

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
########################
# 8. 진입점
########################
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "/koina/src/client/front.py"]