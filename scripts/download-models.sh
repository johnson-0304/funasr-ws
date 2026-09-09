#!/usr/bin/env bash
# Download the sherpa-onnx Fun-ASR-Nano int8 model and the Silero VAD model.
# Usage: scripts/download-models.sh [target-dir]   (default: ./models)
# Set MODEL_MIRROR=modelscope to fetch the ASR model from ModelScope instead of GitHub.
set -euo pipefail

TARGET="${1:-models}"
ASR_NAME="sherpa-onnx-funasr-nano-int8-2025-12-30"
GITHUB_BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
MODELSCOPE_BASE="https://modelscope.cn/models/csukuangfj/asr-models/resolve/master"

case "${MODEL_MIRROR:-github}" in
  github) ASR_URL="${GITHUB_BASE}/${ASR_NAME}.tar.bz2" ;;
  modelscope) ASR_URL="${MODELSCOPE_BASE}/${ASR_NAME}.tar.bz2" ;;
  *) echo "unknown MODEL_MIRROR: ${MODEL_MIRROR}" >&2; exit 1 ;;
esac

mkdir -p "${TARGET}"

if [ ! -f "${TARGET}/silero_vad.onnx" ]; then
  echo "downloading silero_vad.onnx"
  curl -fsSL --retry 5 --retry-delay 3 -o "${TARGET}/silero_vad.onnx" "${GITHUB_BASE}/silero_vad.onnx"
fi

if [ ! -f "${TARGET}/${ASR_NAME}/llm.int8.onnx" ]; then
  echo "downloading ${ASR_NAME} from ${ASR_URL}"
  TMP="$(mktemp -d)"
  curl -fsSL --retry 5 --retry-delay 3 -o "${TMP}/model.tar.bz2" "${ASR_URL}"
  tar xjf "${TMP}/model.tar.bz2" -C "${TARGET}"
  rm -rf "${TMP}"
  # Test wavs are not needed at runtime.
  rm -rf "${TARGET}/${ASR_NAME}/test_wavs"
fi

echo "models ready in ${TARGET}:"
du -sh "${TARGET}/${ASR_NAME}" "${TARGET}/silero_vad.onnx"
