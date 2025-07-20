#!/usr/bin/env bash
# shellcheck disable=SC2164
ROOT=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P)
cd "$ROOT"

function get_tag() {
  TAG="$(git describe --tags)"
  if [[ "$1" = "test" ]]; then
    TAG="$(git describe --tags)-st"
  fi
  if [[ "$TAG" = "" ]]; then
    TAG="not-released"
  fi
  echo $TAG
}

_tag=$(get_tag $1)
echo $_tag
DOCKER_BUILDKIT=1 docker build . -t linky1584/koina:"$_tag"
