#!/usr/bin/env bash
# QMIX + SMAC 3s_vs_5z，8 路并行环境（smoke test / 正式训练均可）
#
# Usage:
#   bash run_qmix_3s_vs_5z.sh
#   GPU=1 T_MAX=2050000 bash run_qmix_3s_vs_5z.sh
#   USE_CUDA=false bash run_qmix_3s_vs_5z.sh

set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-1}"
SEED="${SEED:-0}"
T_MAX="${T_MAX:-100000}"
USE_CUDA="${USE_CUDA:-true}"
USE_TENSORBOARD="${USE_TENSORBOARD:-true}"
PYTHON="${PYTHON:-python3}"

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "QMIX parallel test: map=3s_vs_5z batch_size_run=8 GPU=${GPU} t_max=${T_MAX} seed=${SEED}"

"${PYTHON}" src/main.py --config=qmix --env-config=sc2 with \
  env_args.map_name=3s_vs_5z \
  runner=parallel \
  batch_size_run=8 \
  t_max="${T_MAX}" \
  seed="${SEED}" \
  use_cuda="${USE_CUDA}" \
  use_tensorboard="${USE_TENSORBOARD}"
