# AGENTS.md

## RoutePilot v0.2

执行时遵循 `program.md`。

关键边界：

- RoutePilot 面向通用预测实验，不在 agent/skills 层写死套餐、外卖、单品或固定模型。
- 评测 Agent 工具来自 `skills/`，不要在 `core/` 下新增评测工具。
- `core/` 只处理实验计划、受控修改、训练评测调用、指标对比、trial wrapper 和 rollback。
- LangGraph state 只传 artifact path、小对象和状态标记。
- `main.py run` 只跑一轮 graph；如果 `--experiment` 缺失，可以从 `--ask` 中提取已存在的项目目录。
- `loop.py` 只做多轮外层循环，不重复业务节点逻辑。
- Agent1 只能基于扫描、日志、代码、指标、badcase 等证据生成通用特征实验计划。
- Agent2 必须先复制真实入口和必要 Python 依赖到 `runs/<trial>/code/`，再基于副本生成本轮 `train.py`；原实验源码不被覆盖。
- 具体业务适配器可以存在于 runtime/core 层，但不得污染通用 skills 和两个 agent 的提示/报告逻辑。