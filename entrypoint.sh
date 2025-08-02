#!/usr/bin/env bash
set -e

# 1. conda 환경 활성화
source /opt/conda/etc/profile.d/conda.sh
conda activate mfa

# 2. 헬스체크/사전 점검
mfa version >/dev/null 2>&1 || { echo "MFA not available"; exit 1; }

# 3. exec 로 PID 1 교체
exec "$@"
