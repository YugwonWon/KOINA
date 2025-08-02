#!/usr/bin/env bash
set -euo pipefail

# ─── helper: git 태그 기반 버전 계산 ─────────────────────────────────
function get_tag() {
  GIT_DESCRIBE="$(git describe --tags --long --match "v[0-9]*" 2>/dev/null || echo "")"
  if [[ -z "$GIT_DESCRIBE" ]]; then
    echo "not-released"
    return
  fi

  if [[ "$GIT_DESCRIBE" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)-([0-9]+)-g[0-9a-f]+$ ]]; then
    BASE="${BASH_REMATCH[1]}"
    CNT="${BASH_REMATCH[2]}"
    MAJMIN=$(echo "$BASE" | sed -E 's/^([0-9]+\.[0-9]+)\..*$/\1/')
    TAG="v${MAJMIN}.${CNT}"
  elif [[ "$GIT_DESCRIBE" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    BASE="${BASH_REMATCH[1]}"
    MAJMIN=$(echo "$BASE" | sed -E 's/^([0-9]+\.[0-9]+)\..*$/\1/')
    TAG="v${MAJMIN}.0"
  else
    echo "not-released"
    return
  fi

  local mode="${1:-}"
  if [[ "$mode" = "test" ]]; then
    TAG="${TAG}-st"
  fi

  echo "$TAG"
}

REGISTRY="linky1584/koina"
TAG=$(get_tag)
IMAGE="${REGISTRY}:${TAG}"

echo "→ Building Docker image ${IMAGE}"
docker buildx build \
  --platform linux/amd64 \
  --build-arg MF_ARCH=x86_64 \
  -t "${IMAGE}" \
  . --load

echo "→ Pushing ${IMAGE} to Docker Hub"
docker push "${IMAGE}"

echo "→ Tagging latest"
docker tag "${IMAGE}" "${REGISTRY}:latest"
docker push "${REGISTRY}:latest"

echo "Deployed ${IMAGE}"
