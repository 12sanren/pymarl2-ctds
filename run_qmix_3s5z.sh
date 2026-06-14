#!/usr/bin/env bash
# QMIX + SMAC 3s_vs_5z，8 路并行环境（smoke test / 正式训练均可）
#
# Usage:
#   bash run_qmix_3s_vs_5z.sh
#   GPU=1 T_MAX=2050000 bash run_qmix_3s_vs_5z.sh
#   USE_CUDA=false bash run_qmix_3s_vs_5z.sh

set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
SEED="${SEED:-0}"
T_MAX="${T_MAX:-3000000}"
USE_CUDA="${USE_CUDA:-true}"
USE_TENSORBOARD="${USE_TENSORBOARD:-true}"
PYTHON="${PYTHON:-python3}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "QMIX parallel test: map=3s5z batch_size_run=${BATCH_SIZE_RUN} GPU=${GPU} t_max=${T_MAX} seed=${SEED}"

"${PYTHON}" src/main.py --config=qmix --env-config=sc2 with \
  env_args.map_name=3s5z \
  runner=parallel \
  batch_size_run=${BATCH_SIZE_RUN} \
  t_max="${T_MAX}" \
  seed="${SEED}" \
  use_cuda="${USE_CUDA}" \
  use_tensorboard="${USE_TENSORBOARD}"
