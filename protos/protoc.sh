#!/bin/bash

# Function to compile protobuf files for Python
function protoc_for_python() {
  echo "protoc for python"
  PY=../venv/bin/python3
  mkdir -p ${PY_LIB}

  for p in "${PROTO_SRCS[@]}"; do
    $PY -m grpc_tools.protoc -I. \
      --python_out=${PY_LIB} \
      --mypy_out=${PY_LIB} \
      "$p"
  done

  for p in "${PROTO_SRCS[@]}"; do
    $PY -m grpc_tools.protoc -I. \
      --grpc_python_out=${PY_LIB} \
      --mypy_grpc_out=${PY_LIB} \
      "$p"
  done

  touch "${PY_LIB}/baikal/speech/__init__.py"
}

# Set the current path
current_path=$PWD

# List of protobuf source files
PROTO_SRCS=(
  baikal/speech/stt_service_v1.proto
  baikal/speech/recognition_config.proto
  baikal/speech/stt_service.proto
  baikal/speech/forced_align_service.proto
  baikal/speech/vad_service.proto
)

# Directory for generated Python files
PY_LIB=$(realpath ../src/lib)

# Check if force option is provided
if [[ $1 == "-f" ||  $1 == "-force" ||  $1 == "--f" ||  $1 == "--force" ]]; then
  protoc_for_python
else
  test -d ${PY_LIB} || protoc_for_python
fi
