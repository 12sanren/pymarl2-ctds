# CTDS+ 版本说明

同一套代码，通过 **config 切换** 两种教师变体（无需切分支）。

| 变体 | Config | 教师结构 | 速度 | 推荐地图 |
|------|--------|----------|------|----------|
| **Lite** | `ctds_plus_lite_qmix` | MAPPO `share_obs` + unit cross-attn | ~与 CTDS 接近 | MMM2、大图、优先墙钟时间 |
| **Rel** | `ctds_plus_qmix` | Lite + **agent-level self-attention** | ~CTDS 的 2.5× | 3s5z、需要强协作建模 |

公共部分：MAPPO 式 `get_state_agent` 教师输入、QMIX 教师 TD、NRNN 学生 + Q 蒸馏。

## 运行

```bash
# Rel（默认 3s5z 脚本）
bash run_ctds_plus_3s5z.sh

# Lite（MMM2 脚本已指向 lite）
bash run_ctds_plus_MMM2.sh

# 手动指定
python src/main.py --config=ctds_plus_lite_qmix --env-config=sc2 with env_args.map_name=MMM2 ...
python src/main.py --config=ctds_plus_qmix --env-config=sc2 with env_args.map_name=3s5z ...
```

## 日志路径

```
results/sacred/{map}/{config_name}_{map}_env8_{timestamp}/1/
  config.json   # 完整超参（含 use_agent_relation）
  info.json     # battle_won_mean, test_tea_*, test_stu_*, loss_kd 等
```

TensorBoard：`results/tb_logs/`（体积大，未纳入 git；本地 `use_tensorboard=true` 可生成）。

## 已提交结果索引（2026-06-14）

| 地图 | 变体 | 目录前缀 | 备注 |
|------|------|----------|------|
| MMM2 | Lite | `ctds_plus_qmix_MMM2_env8_2026-06-14_16-31-21` | 无 agent_rel |
| 3s5z | Rel | `ctds_plus_qmix_3s5z_env8_2026-06-14_18-59-52` | `use_agent_relation: true` |
| 3s_vs_5z | Lite | `ctds_plus_qmix_3s_vs_5z_env8_2026-06-14_16-36-47` | |
| 3s_vs_5z | CTDS baseline | `ctds_qmix_3s_vs_5z_env8_2026-06-14_16-41-18` | 对比用 |

查看胜率：各目录下 `info.json` 的 `test_tea_battle_won_mean` / `test_stu_battle_won_mean` 末尾值。

## 给同学复现实验

1. `git clone` 后安装 SMAC / SC2（见仓库 README）
2. 按上表选 config 与 `run_ctds_plus_*.sh`
3. 对比时固定 `seed=0`、`t_max=3000000`、`batch_size_run=8`
