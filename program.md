# RoutePilot 程序规则

## 目标

构建一个面向通用预测实验的最小 two-agent 自动研究循环。

Agent1 扮演数据科学家角色：
- 通过 `skills/` 下的 forecast skills 分析预测误差
- 总结 badcase 模式
- 诊断可能原因，但不声称确定性根因
- 生成可验证的特征假设
- 产出 `experiment_plan.yaml`

Agent2 扮演算法工程师角色：
- 读取 `experiment_plan.yaml`
- 应用允许范围内的特征修改
- 运行 toy fixture 的训练/评测，或为真实预测实验生成 trial 级别的 `train.py` wrapper
- 对比指标
- 使用确定性规则决定 keep 或 rollback

## 场景

v0.2 支持通用预测实验。当前 runtime 可以包含面向已知业务仓库的具体 adapter，但 Agent1、Agent2 和 `skills/` 必须保持业务无关。

对于真实实验，模型家族和目标函数应尽可能从代码、文档、配置或扫描产物中发现。如果无法识别，使用 `unknown`，不要编造值。

MVP 实验方向固定为：
- 特征新增/移除或特征开关实验
- 除非有明确证据支持，否则不修改模型家族
- 可根据实际情况优化 objective/loss、训练参数、正则化、采样策略等
- 不修改 label、metric 或数据切分，除非用户明确批准

不支持：
- 在 skills 或 agents 中写入业务特定 hardcoding
- baseline comparison

## 评测边界

评测 Agent 的能力来自本地 `skills/` 目录。

不要在 `core/` 下重复实现 forecast 评测工具。

以下任务必须使用 skill scripts 或 skill artifacts：
- artifact scanning
- code and log analysis
- metric calculation
- scene metrics
- badcase mining
- forecast report context

## 可编辑边界

toy fixtures 允许修改：
- `benchmark/feature_config.yaml`
- `benchmark/feature_policy.py`
- `benchmark/train.py` 中标记出的 policy block
- `runs/*/experiment_notes.md`

真实预测实验允许生成：
- `runs/*/code/train.py`，从真实入口复制而来，并且只在 trial 目录内 patch
- 从真实 `src/` 目录复制到 `runs/*/code/` 的 Python 依赖
- `runs/*/data/` 下生成的 input manifests
- `runs/*/outputs/real_outputs/` 下生成的真实输出
- `runs/*/logs/` 下生成的日志
- `runs/*/standardized/` 下生成的标准化 artifacts

禁止修改：
- `src/` 下的真实业务源码文件
- `benchmark/evaluate.py`
- metric definitions
- actual data
- data split logic
- `benchmark/train.py` 中标记出的 policy block 之外的代码

## 主指标

WAPE

## 次要指标

- Bias
- MAE
- RMSE
- high_target_wape
- low_target_wape
- underestimation_ratio
- overestimation_ratio

## Keep / Rollback 规则

仅当以下条件全部满足时 keep：
- WAPE 至少改善 `0.005`
- `abs(Bias)` 恶化不超过 `0.02`
- training succeeds
- evaluation succeeds

否则 rollback。

## 护栏

- 不执行 baseline comparison。
- 不修改 actual data。
- 不修改 metric definitions。
- 不覆盖真实业务源码。
- 没有证据时不声称 root cause。
- 不运行无限数量的实验。
- 保持 LangGraph state 足够小：只传 paths、小型 dict 和 status flags。
