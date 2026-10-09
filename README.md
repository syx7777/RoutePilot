# RoutePilot

RoutePilot v0.2，代码包名为 `routepilot`，是一个面向通用预测实验的自动诊断与实验验证 Agent。它读取一个已有预测实验目录中的代码、日志、预测结果和真实值，先用本地 forecast skills 做评测与 badcase 诊断，再生成一轮可验证的特征实验计划，由 trial 级别的训练副本执行实验，最后输出指标对比、keep/rollback 决策和结论报告。

项目目标不是绑定某个业务线、模型或数据形态，而是沉淀一套通用的预测实验研究循环：

```text
已有预测实验
  -> 产物扫描与标准化
  -> 指标评测、场景拆解、badcase 定位
  -> 生成特征假设和实验计划
  -> 复制训练入口到 trial 目录并改造副本
  -> 运行训练与评测
  -> 对比新旧指标并生成报告
```

## 项目结构

```text
.
├── main.py                         # 单轮实验入口
├── loop.py                         # 多轮 trial 循环入口
├── program.md                      # 项目运行规则与边界
├── routepilot/
│   ├── agents/                     # Agent1/Agent2 的计划、决策和报告逻辑
│   ├── core/                       # 实验计划、受控修改、训练调用、指标对比和 rollback
│   ├── orchestrators/              # LangGraph/sequential 编排入口
│   └── runtime/                    # skill 调用、LLM 客户端、产物归档和运行记录
├── skills/                         # forecast 评测、诊断、报告相关 skills
├── tests/                          # 单元测试与 fixture
└── runs/                           # 本地运行输出目录
```

`skills/` 是评测能力的来源，负责扫描、日志/代码分析、指标计算、场景拆解、badcase 挖掘、建议和报告上下文。`core/` 只处理实验执行侧逻辑，包括 schema 校验、受控代码修改、训练评测调用、指标对比、trial wrapper 和 rollback。`runtime/` 负责把技能、LLM、trace、token 用量、运行状态和 trial 归档串起来。

## 两个 Agent

RoutePilot 的一次运行由两个角色协作完成。

Agent1 负责研究和计划。它基于 skill 产物读取证据，形成误差分析、badcase 摘要、问题上下文、特征假设和实验计划。它不会在缺少证据时声称确定性根因，也不会把业务特定字段或模型写死到通用提示中。

Agent2 负责执行和复核。对于真实预测实验，它会先把训练入口和必要 Python 依赖复制到 `runs/<trial>/code/`，再基于副本生成本轮 `train.py` 并运行实验；原始实验源码不会被覆盖。实验结束后，Agent2 会计算新指标、对比旧指标，并按固定规则决定 keep 或 rollback。

## 安装

项目要求 Python 3.11 或以上版本。

```bash
uv sync
```

即可同步环境


## LLM 配置

RoutePilot 通过 OpenAI-compatible 接口调用 LLM。配置可以写入本地 `.env`，也可以在 shell 中导出环境变量；不要把真实 API key 提交到仓库。仓库提供了 `.env.example`，可以复制后按需填写：

```bash
cp .env.example .env
```

豆包示例：

```bash
export DOUBAO_API_KEY="你的豆包 API Key"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export DOUBAO_MODEL="你的 Ark endpoint id"
```

DeepSeek 示例：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-v4-pro"
export DEEPSEEK_THINKING="enabled"
export DEEPSEEK_REASONING_EFFORT="high"
```

通用 OpenAI-compatible provider：

```bash
export ROUTEPILOT_LLM_PROVIDER="openai-compatible"
export LLM_API_KEY="你的 API Key"
export LLM_BASE_URL="https://api.example.com/v1"
export LLM_MODEL="你的模型名"
```

接口模式会按模型名选择：GPT/Codex 系列默认请求 `POST /v1/responses`，GLM、DeepSeek、豆包以及其他非 GPT/Codex 模型默认请求 `POST /v1/chat/completions`。DeepSeek `deepseek-v4-pro` 默认会发送 `thinking` 和 `reasoning_effort` 参数；如需关闭，可设置 `DEEPSEEK_THINKING=disabled`。

如果你的 provider 需要固定某一种接口，可以显式指定：

```bash
export ROUTEPILOT_LLM_API_MODE="chat"
# 或
export ROUTEPILOT_LLM_API_MODE="responses"
```

如果已经在 `.env` 或 shell 中设置了 `ROUTEPILOT_LLM_PROVIDER` 和 `LLM_MODEL`，运行命令可以省略 `--llm-provider` 和 `--model`，系统会按环境变量选择模型。

也可以只传 `--model deepseek-v4-pro` 或 `--model gpt-...`，系统会根据模型名前缀推断 provider。未显式配置时，默认 provider 为 `doubao`。

LLM 请求默认会对临时网络、代理、超时以及 `429/500/502/503/504` 进行重试：

```bash
export ROUTEPILOT_LLM_RETRIES=3
export ROUTEPILOT_LLM_RETRY_BACKOFF_SECONDS=5
export ROUTEPILOT_LLM_RETRY_MAX_BACKOFF_SECONDS=60
```

设置 `ROUTEPILOT_LLM_RETRIES=0` 可以关闭重试。

## 单轮运行

最常用入口是 `main.py run`。`--experiment` 指向已有预测实验目录，`--ask` 描述本轮目标，`--output` 指定 trial 输出目录。

```bash
python main.py run \
  --experiment /path/to/forecast_experiment \
  --ask "分析预测误差，提出一个特征实验并验证效果" \
  --llm-provider deepseek \
  --model deepseek-v4-pro \
  --orchestrator langgraph \
  --output runs/trial_001
```

如果 `--ask` 中已经包含一个存在的项目目录，也可以省略 `--experiment`：

```bash
python main.py run \
  --ask "项目地址 /path/to/forecast_experiment，分析预测误差，提出一个特征实验并验证效果" \
  --output runs/trial_001
```

运行完成后，终端会打印 decision、主指标、WAPE/Bias 前后对比、最终报告路径、trace 路径和 token 用量。

`--orchestrator` 当前支持 `langgraph` 和 `sequential`。`sequential` 入口复用同一轮 graph，主要用于 CLI 参数兼容和后续扩展；默认使用 `langgraph`。

如果运行失败，CLI 会打印异常信息；若已写出错误产物，可在 trial 目录查看 `error_report.md` 和 `error_report.json`。

## 多轮运行

`loop.py` 用于连续执行多个 trial。它只负责 trial 目录管理、调用单轮 graph、汇总历史结果和选择最佳 trial，不重复实现业务节点逻辑。

```bash
python loop.py \
  --experiment /path/to/forecast_experiment \
  --ask "基于当前评测结果自动尝试特征构建优化并输出最终效果" \
  --max-trials 3 \
  --orchestrator langgraph \
  --output runs/
```

多轮运行会生成：

- `runs/trial_001/`、`runs/trial_002/` 等独立 trial 目录
- `runs/run_history.csv`：每轮 decision、主指标和报告路径
- `runs/best_trial_review.md` / `runs/best_trial_review.json`：最佳 trial 选择依据
- `runs/final_report.md`：多轮汇总报告

多轮运行最多执行 `--max-trials` 次；如果 Agent1 产出的候选实验数量少于 `--max-trials`，循环会按候选数量提前停止。

## 运行产物

单轮 trial 结束后，产物会按用途整理到分组目录中；`final_report.md` 和 `artifact_index.md` 会在 trial 根目录保留入口副本，方便快速查看。

```text
runs/trial_001/
├── final_report.md
├── artifact_index.md
├── data/
│   └── input_manifest.json
├── outputs/
│   └── real_outputs/
├── agent1/
│   ├── artifact_contract.json
│   ├── analysis_report.md
│   ├── problem_context.json
│   ├── badcase_diagnosis.md
│   ├── feature_hypothesis.yaml
│   ├── candidate_experiments.yaml
│   ├── experiment_plan.yaml
│   └── agent1_program.md
├── agent2/
│   ├── agent2_execution_plan.yaml
│   ├── source_evaluation_context.json
│   ├── output_contract.json
│   ├── run_status.json
│   ├── review_result.json
│   ├── experiment_review.md
│   └── code/
│       └── train.py
├── evaluation/
│   ├── metrics.json
│   ├── metrics_summary.csv
│   ├── new_metrics.json
│   ├── new_metrics_summary.csv
│   ├── metric_comparison.json
│   ├── scene_metrics.csv
│   ├── badcases.csv
│   ├── badcase_summary.json
│   ├── column_mapping.json
│   └── anomaly_summary.json
├── reports/
│   ├── forecast_report.md
│   ├── optimization_suggestions.md
│   ├── report_context.json
│   └── final_report_context.json
├── standardized/
│   ├── standardized_prediction.csv
│   └── standardized_actual.csv
├── logs/
│   ├── train.log
│   └── eval.log
├── audit/
│   ├── agent_status.json
│   ├── agent_timeline.jsonl
│   ├── llm_calls.jsonl
│   ├── token_usage.json
│   ├── trace.jsonl
│   ├── scan_result.json
│   ├── artifact_summary.json
│   ├── code_analysis.json
│   └── log_summary.json
└── archive/
    ├── archive_manifest.json
    ├── inputs/
    ├── outputs/
    ├── agent1/
    ├── agent2/
    ├── evaluation/
    ├── reports/
    ├── logs/
    └── audit/
```

`reports/forecast_report.md` 是完整评测诊断报告，包含扫描、指标、场景拆解和 badcase 证据。`final_report.md` 是实验验证结论报告，重点回答本轮实验是否有效、指标为何改善或失败、当前应该 keep 还是 rollback，以及下一轮应优先尝试什么。`archive/` 会按分组创建归档副本或符号链接，便于把单轮 trial 作为完整证据包迁移。

## 输入要求

预测实验目录应尽量包含以下信息：

- 预测结果 CSV 和真实值 CSV，或同一 CSV 中同时包含 prediction 与 actual 列
- 训练入口 Python 文件
- 可选的训练日志或评测日志
- 可选的配置文件、文档、metrics 文件和历史输出

RoutePilot 会先扫描目录，再由 Agent1 基于证据选择 `prediction_path`、`actual_path`、列名、测试窗口、训练入口和可复用字段，生成 `artifact_contract.json`。如果模型家族、目标函数或业务含义无法从代码、文档或配置中识别，会标记为 `unknown`，不会编造。

输入可以是 toy fixture，也可以是真实预测实验。真实实验中，Agent2 会先复制训练入口和必要 Python 依赖到 trial 目录，再只修改副本中的本轮 `train.py` wrapper。

## 决策规则

当前 keep/rollback 使用固定阈值：

- WAPE 至少改善 `0.005`
- `abs(Bias)` 恶化不超过 `0.02`
- 训练成功
- 评测成功

只要任一条件不满足，本轮实验就 rollback。真实预测实验会在 trial 副本中运行；rollback 表示本轮方案不进入最佳方案，不会覆盖原始业务源码。

## 安全边界

RoutePilot 保持通用预测实验边界：

- 不在 `skills/`、Agent prompt 或报告逻辑中写死具体业务线、套餐、外卖、单品或固定模型
- 不在 `core/` 下新增评测工具，评测能力来自 `skills/`
- 不修改 actual 数据、指标定义或数据切分逻辑
- 不覆盖真实实验目录下的原始 `src/*` 或训练入口源码
- 不执行无限实验，多轮数量由 `--max-trials` 控制
- LangGraph state 只传 artifact path、小对象和状态标记，不传大 CSV、DataFrame 或完整 prompt
- Agent1 只能基于扫描、日志、代码、指标、badcase 等证据生成通用特征实验计划
- Agent2 必须在 trial 副本中执行修改和训练，不覆盖原实验源码

真实预测实验允许生成的内容集中在 trial 目录：

- `runs/<trial>/code/train.py`
- `runs/<trial>/code/*.py` 依赖副本
- `runs/<trial>/data/input_manifest.json`
- `runs/<trial>/outputs/real_outputs/*`
- `runs/<trial>/logs/train.log`
- `runs/<trial>/logs/eval.log`
- `runs/<trial>/standardized/*.csv`

## 开发与测试

运行单元测试：

```bash
pytest
```

运行指定测试：

```bash
pytest tests/test_langgraph_flow.py
```

`program.md` 是项目规则的权威说明；修改 Agent 行为、评测边界或运行约束时，应同步更新 `program.md` 和 README。
