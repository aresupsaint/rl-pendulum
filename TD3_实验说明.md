# 单杆与双杆倒立摆：TD3 实验说明

## 任务定义

| 环境 | 控制目标 | 动作 | 成功指标 |
|---|---|---|---|
| `InvertedPendulum-v5` | 小车上的单杆持续直立 | 连续推力 | 未跌倒且存活至 1000 步时间上限 |
| `InvertedDoublePendulum-v5` | 小车上的双杆持续直立 | 连续推力 | 未跌倒且存活至 1000 步时间上限 |
| `Pendulum-v1` | 单杆从随机姿态摆起并稳定 | 连续力矩 | 末 50 步中至少 80% 的时间处于自定义直立区域 |

`Acrobot-v1` 是双杆摆起任务，达到高度目标便结束；它不能验证持续平衡。保留原 `rl_experiments.py` 作为 PPO/SAC 摆起对照。

## 与原代码相比的改进

1. 明确区分摆起和平衡；选择与控制目标一致的环境和成功指标。
2. 连续控制使用 TD3：双 Q 网络抑制价值高估，目标动作平滑降低对局部错误峰值的追逐，actor 延迟更新减少训练抖动。
3. 所有训练按环境步数设定预算，避免不同环境的回合长度使训练量不可比。
4. 经验池使用预分配数组；动作统一规范到 `[-1,1]`，critic 接收规范化动作，再映射到环境动作范围。
5. 只在 `terminated` 时停止 Bellman 引导；`truncated` 是时间上限，继续用末状态价值引导。
6. 训练与评估使用独立环境和固定评估种子，记录平均回报、样本标准差、平均存活步数及成功率。
7. 支持多个随机种子，保存每个种子的 CSV、摘要、最佳及最后策略权重，并汇总跨种子回报。

## 安装与运行

以下命令假设已进入本目录，Python 3.12 已可用：

```powershell
python -m pip install -r requirements_td3.txt
python td3_inverted_pendulums.py --env InvertedPendulum-v5 --steps 200000 --seeds 0 1 2 3 4 --eval-episodes 20 --holdout-episodes 100 --out results_td3
python td3_inverted_pendulums.py --env InvertedDoublePendulum-v5 --steps 500000 --seeds 0 1 2 3 4 --eval-episodes 20 --holdout-episodes 100 --out results_td3
python td3_inverted_pendulums.py --env Pendulum-v1 --steps 200000 --seeds 0 1 2 3 4 --eval-episodes 20 --holdout-episodes 100 --out results_td3
```

步数是实验预算建议，不保证收敛。先看 `eval.csv` 中的学习曲线，再增加步数或调整超参数。单杆和双杆平衡环境需要 MuJoCo；未安装时可先运行 `Pendulum-v1` 检查代码路径。

## 输出

每个 `环境/seedN` 目录包含：

- `train.csv`：完整训练回合的回报、长度及结束类型。
- `eval.csv`：每次评估的平均回报、标准差、平均长度、成功率和直立比例。
- `summary.json`：环境、种子、步数、更新次数、版本、参数及最终评估。
- `best.pt`：按评估平均回报选出的最佳策略。
- `last.pt`：最后一步策略。
- `resume.pt`：在回合边界自动保存的完整训练状态，可续训。
- `holdout.json`、`holdout_episodes.csv`：用独立测试种子评估最佳策略的汇总与逐回合数据。
- `baselines.json`、`baseline_episodes.csv`：Pendulum-v1 上随机动作与能量摆起加 PD 稳定控制器的对照结果。
- `aggregate.json`：多训练种子的最终测试均值、标准差和近似 95% 置信区间。

`best.pt` 和 `last.pt` 保存用于推理的 actor 权重与动作范围。`resume.pt` 保存 actor/critic、优化器、经验池及随机数状态；快照只在完整回合结束后写入，意外中断时从最近快照继续，可能重做少量环境步。跨种子置信区间以训练种子为统计单位并使用 t 区间；单个策略的回合级置信区间采用正态近似，建议结合逐回合 CSV 一起报告。GPU 计算在不同硬件或驱动上可能无法逐位复现。

Pendulum-v1 的 PD 对照采用能量整形完成摆起，并在接近直立时切换到 PD 阻尼控制；这是一个手工控制基线，不是另一种强化学习算法。其参数是可调的，报告结果时应连同控制规则一起说明。

若训练中断，使用对应目录中的续训快照并把 `--steps` 设为目标总环境步数：

```powershell
python td3_inverted_pendulums.py --env Pendulum-v1 --steps 200000 --seeds 0 --resume results_td3/Pendulum-v1/seed0/resume.pt --out results_td3
```

独立测试示例：

```powershell
python evaluate_td3.py results_td3/InvertedPendulum-v5/seed0/best.pt --episodes 20 --seed 300000 --out results_td3/InvertedPendulum-v5/seed0/holdout.json
```

## 本次实际运行

在 CPU、PyTorch 2.14.1、Gymnasium 1.3.0、MuJoCo 3.14.0、种子 0 下，训练后以**未参与选最佳模型**的 20 个测试种子评估：

| 环境 | 训练步数 | 最佳权重来自第几步 | 独立测试平均回报 | 平均长度 | 成功回合 |
|---|---:|---:|---:|---:|---:|
| `InvertedPendulum-v5` | 50,000 | 30,000 | 1000.00 | 1000 | 20/20 |
| `InvertedDoublePendulum-v5` | 100,000 | 30,000 | 9359.82 | 1000 | 20/20 |
| `Pendulum-v1` | 50,000 | 50,000 | −150.78 | 200 | 20/20（自定义末段直立指标） |

Pendulum 独立测试的整体直立时间比例为 0.847。三项成绩各只有**一个训练种子**，不能代表跨种子的稳定性能，也不能直接与原 DDPG 的 6 回合短跑比较。对结果有正式要求时，应执行上面的多种子命令，并报告每个种子的最终测试结果。
