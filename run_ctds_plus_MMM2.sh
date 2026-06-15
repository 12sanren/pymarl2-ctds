#!/usr/bin/env bash
# CTDS+ Lite + QMIX + SMAC MMM2（无 agent relation，较快）
#
# Usage:
#   bash run_ctds_plus_3s5z.sh
#   GPU=1 T_MAX=2050000 bash run_ctds_plus_3s5z.sh
#   USE_CUDA=false bash run_ctds_plus_3s5z.sh

set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-2}"
SEED="${SEED:-0}"
T_MAX="${T_MAX:-3000000}"
USE_CUDA="${USE_CUDA:-true}"
USE_TENSORBOARD="${USE_TENSORBOARD:-true}"
PYTHON="${PYTHON:-python3}"

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "CTDS+ Lite: map=MMM2 batch_size_run=8 GPU=${GPU} t_max=${T_MAX} seed=${SEED}"

"${PYTHON}" src/main.py --config=ctds_plus_lite_qmix --env-config=sc2 with \
  env_args.map_name=MMM2 \
  runner=parallel \
  batch_size_run=8 \
  t_max="${T_MAX}" \
  seed="${SEED}" \
  use_cuda="${USE_CUDA}" \
  use_tensorboard="${USE_TENSORBOARD}"
