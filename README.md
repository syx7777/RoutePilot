# 🚀 RoutePilot

给 RoutePilot 一个**已经能跑的预测实验**（有训练入口、预测结果、真实值），它会读你的代码和日志做评测诊断，提出几条特征实验，在副本里改代码、重跑、算新指标，最后告诉你这次改动该 **keep 还是 rollback**，并写一份报告。

## 🌟 核心优势

- 🔌 **不改代码就能接入**：换项目只写一份 `routepilot.yaml`，用 `init / validate / run` 三个命令跑起来，框架本身零改动。
- 🛡️ **安全隔离，随时可回滚**：Agent 只能改 `editable` 白名单里的文件，`protected` 里的入口源码一律不碰；keep 就地生效，但原始内容会留档到 `revert/`，rollback 直接按快照恢复。
- 🔍 **证据驱动，不编造**：模型家族、目标函数这类信息，只有代码或文档能证明才会填，否则标 `unknown`。
- 💰 **成本看得见**：按 step 动态选模型档位（`strong / weak / static / dynamic`），每次调用都记账，输出成本、tokens、P50/P95 和成功率。
- 📊 **瓶颈自动归因**：内置 Profiler 把耗时拆成 `project_run / llm_inference / evaluate` 三类，谁占比高就把路由往哪调，形成闭环。
- 🔁 **实验可复现**：支持 `--proposals` 离线提案，把"优化闭环"和"大模型随机性"分开，同一份提案能重复跑出同一结果。
- 🧱 **通用，不绑业务**：指标驱动（`wape / bias / mae / rmse / mape / smape`），时序预测、宽表回归都能接。

## 🎯 适用场景与目标用户

**✅ 适合谁**

- 算法工程师、数据科学家：手上已经有能跑的预测实验，想把"看 badcase → 想特征 → 跑实验 → 比指标"这套流程自动化。
- 预测建模团队：要在多个项目上复用同一套自动优化流程，又不想为每个项目改框架代码。
- 关注推理成本的人：希望在质量不掉的前提下，把 LLM 开销压下来。

**📌 典型场景**

- 需求 / 销量 / 时序预测的特征迭代：自动提出特征实验，并验证它是否真的有效。
- badcase 诊断：定位误差集中在哪些场景，为下一轮优化提供证据。
- 成本与耗时优化：用不同路由模式对比各模型档位的成本和延迟。
- 多项目接入：一套流程同时接入两个不同的预测项目。

**🚫 不适合**

- 没有能跑的训练 / 预测入口。
- 指标口径不稳定（同配置重跑结果差异大）且无法把噪声收敛下来。
- 改动面无法回滚。

## ⚡ 跑通 demo

前提：Python 3.11+，装好 [uv](https://docs.astral.sh/uv/)。

**1）装依赖**

```bash
uv sync
```

**2）跑 demo（不用配 API key）**

`examples/demo_forecast/` 是一个能跑的最小预测项目，配了一份离线提案，所以全程不调用大模型：

```bash
uv run routepilot validate --manifest examples/demo_forecast/routepilot.yaml
uv run routepilot run --manifest examples/demo_forecast/routepilot.yaml \
  --goal "降低 WAPE" \
  --proposals examples/demo_forecast/proposals.yaml \
  --output runs/demo
```

**3）看结果**

终端会打印每轮结果，大致长这样：

```text
[baseline] wape=0.7370
[trial 1] keep: wape 改善 0.7228 ≥ 0.01 | wape=0.0142
[trial 2] rollback: wape 改善 -0.4682 < 0.01 | wape=0.4824
```

## 🔌 接到你自己的项目

三步：`init` 生成配置，`validate` 检查，`run` 跑起来。

```bash
uv run routepilot init --root ../my_forecast_project
uv run routepilot validate --manifest ../my_forecast_project/routepilot.yaml
uv run routepilot run --manifest ../my_forecast_project/routepilot.yaml --goal "降低 WAPE"
```

- `init` 只做"接入前能判定"的推断，没把握的会写进 `questions` 让你人工确认。它靠已有的 `outputs/prediction.csv` 这类文件来认列名，所以你的项目最好**至少跑出过一次预测结果**再执行 `init`。
- `run` 默认要调大模型。不想用大模型，就加 `--proposals` 传离线提案（和 demo 一样）。
- 你的项目得满足：有能跑的入口、指标口径稳定可算、改动面可回滚。

### routepilot.yaml 里重点看这几项

```yaml
project:
  entrypoint: src/train.py           # 训练入口
run:
  command: [python, src/train.py, --config, configs/base.yaml]   # 启动命令（argv 列表）
artifacts:
  prediction: outputs/prediction.csv
  actual:     data/actual.csv
  columns: {prediction: yhat, actual: y, date: ds}   # 对齐键，date 与 id 至少指定一个
metrics:
  primary: {name: wape, direction: minimize, min_delta: 0.01}   # 主指标与 keep 阈值
  guards:  [{name: bias, max_regression: 0.05}]                 # 护栏指标与最大允许恶化
editable:  [configs/*.yaml]            # 允许修改的文件白名单
protected: [data/**, outputs/**, src/train.py]   # 禁止修改的文件清单，优先级高于 editable
```

keep / rollback 完全看 `metrics`：主指标改善达到 `min_delta`、且护栏没越界，才 keep。

## 🤖 想用真实大模型

把 `.env.example` 复制成 `.env`，填一种 provider 就行：

```bash
cp .env.example .env
```

DeepSeek：

```bash
DEEPSEEK_API_KEY=你的 key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

豆包（Ark）：

```bash
DOUBAO_API_KEY=你的 key
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL=你的 endpoint id
```

不指定时默认走豆包；也可以用 `ROUTEPILOT_LLM_PROVIDER` 显式选一个（比如 `deepseek`）。真实 key 别提交进仓库。

## 🧩 另外两个入口（可选）

- `main.py`：单轮双 Agent 跑一次，适合接到具体业务实验里。

  ```bash
  uv run python main.py run --experiment ../my_forecast_project \
    --ask "分析预测误差，提一个特征实验并验证效果" --output runs/trial_001
  ```

- `loop.py`：多轮循环，跑完自动汇总并挑出最好的一轮。

  ```bash
  uv run python loop.py --experiment ../my_forecast_project \
    --ask "自动尝试特征优化并给出最终效果" --max-trials 3 --output runs/
  ```

## 📄 License

本项目基于 [MIT License](LICENSE) 开源。