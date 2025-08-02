########################
# 1. Base Image + APT
########################
# 공통 변수
ARG MFA_ROOT_DIR=/opt/mfa_models
ARG MF_ARCH=x86_64
ARG MF_VER=25.3.0-3

FROM ubuntu:22.04

ARG MF_ARCH
ARG MFA_ROOT_DIR
ARG MF_URL
ARG MF_VER

ENV CONDA_DIR=/opt/conda \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PATH=/opt/conda/envs/mfa/bin:/opt/conda/bin:$PATH \
    MFA_ROOT_DIR=${MFA_ROOT_DIR}

########################
# 2. APT packages
########################
# ca-certificates 를 먼저 깔고 update-ca-certificates 실행
RUN apt-get update && \
    apt-get install -y sox libsox-fmt-all && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg2 && \
    update-ca-certificates && \
    apt-get install -y --no-install-recommends \
        tzdata ffmpeg libsndfile1 fontconfig fonts-nanum && \
    rm -rf /var/lib/apt/lists/*

# ── i386 라이브러리 설치 ──
RUN dpkg --add-architecture i386 && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libsndfile1:i386 libavcodec-extra:i386 && \
    rm -rf /var/lib/apt/lists/*

########################
# 2. Miniforge + MFA(3.2.1) 설치  (root)
########################
ARG MF_URL=https://github.com/conda-forge/miniforge/releases/download/${MF_VER}/Miniforge3-${MF_VER}-Linux-${MF_ARCH}.sh
RUN curl -fsSL "$MF_URL" -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR" && \
    rm /tmp/miniforge.sh && \
    "$CONDA_DIR/bin/conda" clean -afy

# MFA 전용 env 생성
RUN ${CONDA_DIR}/bin/conda config --set auto_activate_base false && \
    ${CONDA_DIR}/bin/conda config --add channels conda-forge && \
    ${CONDA_DIR}/bin/conda config --set channel_priority strict && \
    ${CONDA_DIR}/bin/conda create -y -n mfa python=3.10 && \
    ${CONDA_DIR}/bin/conda run -n mfa pip install --upgrade pip

# MFA + 필수 라이브러리 설치 (conda/pip 혼합)
RUN ${CONDA_DIR}/bin/conda run -n mfa conda install -y \
        montreal-forced-aligner==3.2.1 \
        numpy pandas scipy matplotlib tqdm \
        textgrid pydub ffmpeg-python && \
    ${CONDA_DIR}/bin/conda run -n mfa pip install joblib==1.1.0 && \
    ${CONDA_DIR}/bin/conda run -n mfa pip install gradio soundfile==0.12.1 praat-parselmouth
RUN /opt/conda/bin/conda run -n mfa pip install \
      python-mecab-ko jamo

########################
# 3. 일반 사용자 생성 + 모델 디렉터리 준비  (root)
########################
ARG MFA_ROOT_DIR
RUN useradd -ms /bin/bash mfauser && \
    mkdir -p ${MFA_ROOT_DIR} && \
    chown -R mfauser:mfauser ${MFA_ROOT_DIR}

########################
# 4. 코드·엔트리포인트 복사
########################
WORKDIR /koina
ENV PYTHONPATH=/koina/src
COPY tests/ /koina/tests/
COPY --chown=mfauser:mfauser src/ ./src/
COPY --chown=mfauser:mfauser entrypoint.sh /entrypoint.sh
RUN chown -R mfauser:mfauser /koina
RUN chmod +x /entrypoint.sh

########################
# 5. 모델 다운로드  (mfauser)
########################
USER mfauser
RUN /opt/conda/bin/conda run -n mfa mfa model download dictionary korean_mfa && \
    /opt/conda/bin/conda run -n mfa mfa model download acoustic   korean_mfa

########################
# 6. conda 캐시·작업 디렉터리
########################
ENV WORKDATA=/home/mfauser/data
RUN mkdir -p $WORKDATA /home/mfauser/.conda/pkgs
ENV CONDA_PKGS_DIRS=/home/mfauser/.conda/pkgs

########################
# 7. 실행
########################
EXPOSE 40080
ENV HOME=/home/mfauser
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "/koina/src/client/front.py"]