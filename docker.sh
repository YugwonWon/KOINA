#!/usr/bin/env bash
# shellcheck disable=SC2164
ROOT=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P)
cd "$ROOT"

function get_tag() {
  # git describe --tags 출력 예: v0.5.0-31-ge9baed6
  # --long: 커밋 해시까지 포함 (나중에 파싱을 위해)
  # --match "v[0-9]*": 'v'로 시작하고 숫자가 뒤따르는 태그만 고려
  GIT_DESCRIBE_OUTPUT="$(git describe --tags --long --match "v[0-9]*" 2>/dev/null)"

  if [[ -z "$GIT_DESCRIBE_OUTPUT" ]]; then
    # 태그가 없는 경우 (예: 초기 커밋)
    echo "not-released"
    return
  fi

  # 출력에서 태그와 커밋 수를 파싱
  # 예: v0.5.0-31-ge9baed6
  # BASH_REMATCH 배열에 캡처 그룹이 저장됩니다.
  if [[ "$GIT_DESCRIBE_OUTPUT" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)-([0-9]+)-g[0-9a-f]+$ ]]; then
    BASE_VERSION="${BASH_REMATCH[1]}" # v0.5.0
    COMMITS_SINCE_TAG="${BASH_REMATCH[2]}" # 커밋 수

    MAJOR_MINOR_VERSION=$(echo "$BASE_VERSION" | sed -E 's/^([0-9]+\.[0-9]+)\..*$/\1/')

    # 최종 TAG 형식: v.MAJOR.MINOR.COMMITS_SINCE_TAG (예: v.0.5.31)
    TAG="v${MAJOR_MINOR_VERSION}.${COMMITS_SINCE_TAG}"
  else
    # 태그는 있지만 파싱 패턴과 일치하지 않는 경우
    # 예: v1.0 (커밋 수가 없는 태그)
    if [[ "$GIT_DESCRIBE_OUTPUT" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
        # 커밋 수가 없는 순수 태그인 경우 (예: v0.5.0)
        BASE_VERSION="${BASH_REMATCH[1]}"
        MAJOR_MINOR_VERSION=$(echo "$BASE_VERSION" | sed -E 's/^([0-9]+\.[0-9]+)\..*$/\1/')
        TAG="v${MAJOR_MINOR_VERSION}.0" # 순수 태그의 경우 커밋 수를 0으로 간주
    else
        echo "not-released" # 예상치 못한 태그 형식
        return
    fi
  fi

  if [[ "$1" = "test" ]]; then
    TAG="${TAG}-st"
  fi
  echo "$TAG"
}

_tag=$(get_tag "$1")
echo "$_tag"
DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 -t linky1584/koina:"$_tag" . --load