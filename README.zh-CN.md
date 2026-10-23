# ThermoResidual-RL

**面向模型失配下地热发电与制冷协同控制的物理引导残差强化学习。**

[English](README.md) | [实验协议](docs/PROTOCOL.md) | [数据说明](docs/DATA.md) | [复现说明](docs/REPRODUCIBILITY.md)

本仓库提供论文所用的守恒随机源—负荷模拟器、六种控制器、训练实现、冻结模型和模拟示例。
下载后无需取得项目现场原始数据，即可运行控制比较与模拟数据生成。

## 方法与控制器

ThermoResidual-RL 将名义热力学分配作为动作参考，根据近期运行历史学习动态修正。
模拟器通过质量和能量平衡耦合地热取热、氨库存、发电、吸附制冷及冷凝回液；
相关随机过程描述有效液压阻力和换热参数的波动。

| 命令名称 | 控制器 |
|---|---|
| `rrl` | ThermoResidual-RL：热力学分配加学习残差 |
| `sac` | 信息对齐的全动作软演员—评论家方法 |
| `pi` | Physics+PI：相同分配器加比例积分反馈 |
| `gspid` | 增益调度比例积分微分控制，含抗积分饱和 |
| `mpc` | 使用名义模型和 OSQP 的约束模型预测控制 |
| `physics` | 名义热力学分配 |

RRL 和 SAC 共用观测、目标预览、网络结构、初始化、优化设置、交互预算和硬命令投影。
二者的区别是动作组合方式；两种动作映射均覆盖完整可行支路流量范围。
主比较不使用在线模型辅助细化。各控制器只使用自身闭环已经到达的测量。
相同过程随机种子对应相同参数扰动流，而不是强制相同输出。

## 安装

已测试环境：Linux、Python 3.11。冻结策略评价可以使用 CPU；CUDA 用于加速训练。
物理环境仍由 CPU 求解。在仓库根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖文件使用不带本机 CUDA 后缀的 `torch==2.6.0`。如使用 GPU，请在环境中安装与设备
匹配的 PyTorch 构建；实际验收环境记录在 `environment-tested.json`。
尚未进行 Windows/macOS 的执行验收。

## 快速运行

```bash
# 仅校验公开文件，不导入 PyTorch
python reproduce.py --verify-only

# 六种控制器，每条 256 步
python reproduce.py --steps 256 --workers 2

# 绘制输出、误差与命令，导出 PDF/SVG/PNG
python plot_results.py --input outputs/demo_256
```

控制间隔为 10 s，前 200 步为预热。因此短示例的误差仅在后 56 步计算，用于检查程序链路，
不等于论文完整评价。输出位于 `outputs/demo_256/`，包括逐点 CSV、结果 JSON、
`metrics.csv` 和 `run_status.json`。重复运行仅在配置和轨迹校验和一致时复用已有结果。

```bash
# 单个算法
python reproduce.py --controllers rrl --steps 256

# 生成新的模拟边界数据，不使用现场文件
python generate_synthetic_data.py --steps 512 --seed 1729
```

`data/synthetic/` 已附带一份生成示例及元数据。压力、源侧热量、冷凝条件和流量来自
同一次闭环计算，明确标为模拟数据，不是现场数据的脱敏副本，也不是独立于动作的取热回放。

## 完整评价

```bash
# 一条完整时长轨迹：3200 步，预热 200 步
python reproduce.py --suite main --controllers rrl --limit 1

# 六算法完整矩阵：480 条
python reproduce.py --suite main --workers 4
python summarize_results.py
```

矩阵为六种控制器 × 四个目标族 × 四种工况 × 五个过程随机实现。
目标族为 `standard`、`medium_unseen`、`very_fast_unseen`、`mixed_step_ramp`。
修订环境低幅值工况为发电 13–15 kW、制冷 14–18 kW，高幅值为发电 12–16 kW、制冷 12–20 kW。
不要混用旧环境的目标范围。

可用 `--profile`、`--condition`、`--controllers` 和 `--limit` 过滤任务。
`--limit` 按清单顺序取任务，不是随机抽样。绘图时应将同一目标族、工况、过程种子下的
各算法 CSV/JSON 放在同一目录；绘图程序拒绝混合不匹配的轨迹。
运行耗时随 CPU、线程和负载变化，不承诺统一时间。

## 从头训练

公开初始状态为成对的未训练权重、空 replay buffer 和零次优化更新，并包含冻结归一化参数；
其中没有现场轨迹。训练命令默认使用正式的 768,000 次交互预算：

```bash
# 小规模链路验收
python train.py --controller rrl --smoke --output outputs/train_smoke_rrl
python train.py --controller sac --smoke --output outputs/train_smoke_sac

# 正式预算，可在两块卡上分别启动
python train.py --controller rrl --device cuda:0 --output outputs/train_rrl
python train.py --controller sac --device cuda:1 --output outputs/train_sac
```

正式训练每个算法使用 16 个 CPU 环境进程，并另启开发评价进程，请按计算资源控制并发。
`--device cpu` 也可训练，但更慢。`--smoke` 使用 1,024 次交互和四个 256 步开发评价，
只验收接口。正式训练保留原开发窗口和共同优化设置。输出目录必须不存在，避免覆盖实验。
`--weight-ratio 0.5` 或 `--weight-ratio 2` 可选择已冻结的奖励权重对照。

全部超参数、观测组成、目标函数和 checkpoint 选择规则见 `docs/PROTOCOL.md`。
过程随机实现用于评价冻结策略，不能作为独立训练初始化的数量。

## 传统基线重新整定

```bash
python tune_baselines.py --controller pi --smoke --output outputs/pi_check
python tune_baselines.py --controller pi --workers 4 --output outputs/pi_grid
python tune_baselines.py --controller mpc --workers 4 --output outputs/mpc_grid
python tune_baselines.py --controller gspid --workers 4 --output outputs/gspid_grid
```

完整搜索分别含 24 组 PI、9 组 MPC、54 组 GS-PID 配置，每组在共同的 12 个开发窗口评价。
以满足验收条件的最低共同综合代价选参，不使用最终评价选参。
新选择保存为 `selected.json`，不会自动覆盖仓库冻结结果；应用方式见复现说明。

## 其他冻结评价

```bash
python reproduce.py --suite stress --limit 1
python reproduce.py --suite weights --limit 1
python reproduce.py --suite sensor --limit 1
```

- `stress`：840 条任务，含共同参考场景和六种工程参数敏感性场景。
- `weights`：320 条任务，对应两种权重比下的四个新增策略。
- `sensor`：984 条历史跨采样诊断任务，将压力采样改为 60 s、控制仍为 10 s；
  包含两种历史压力偏移设置，其中零偏移为 792 条。这不是正常 10 s 场景，
  历史偏移也不是当前绝压数据的推荐换算。完整保留诊断结果，不选择性删除。

后两类压力诊断和参数扰动评价采用原五种控制器，不包含后加的 GS-PID。
跨采样诊断不代表已经验证缺测部署可靠性。

## 文件结构与数据范围

| 目录/文件 | 内容 |
|---|---|
| `source_coupled_revision/` | 守恒模拟器、测量接口、控制器和学习 |
| `experiments/` | 必要训练子类、冻结参数、初始状态、策略与压力预测权重 |
| `configs/` | 示例与完整评价任务清单 |
| `data/synthetic/` | 标记为模拟的示例及元数据 |
| `results/` | 模拟汇总指标和训练历史 |
| `docs/` | 协议、数据字典与复现说明 |
| `tests/` | 因果性、动作权限、代价和守恒测试 |
| `tools/` | 公开文件与模型资产检查 |

部分实验目录保留原标识，以保持配置与资产的一致对应。未包含现场 CSV/Excel、原始日志、
项目报告、论文全文、账号密码、缓存和改稿工具。
公开的数值参数和学习权重属于派生研究资产，不等于原始观测。
原压力预测拟合和现场标定需要受控原始记录；模拟控制与示例生成无需这些数据。

模型为同井文献和现场统计支持的控制导向参考模型，并非黄文博等完整轴向求解器的逐式复制。
守恒验证、现场统计对照和控制效果分别对应不同证据。

## 测试、校验与许可证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/audit_public_release.py
```

公开导出将本地绝对路径改为仓库相对路径，同步更新 checkpoint 的协议元数据。
学习张量不变；原始/公开哈希与张量检查记录见 `PUBLIC_EXPORT.json`，
文件完整性清单见 `RELEASE_MANIFEST.json`。
校验和用于检查变化，并不为任意第三方下载提供身份认证，请只加载可信模型。

本项目代码、文档及项目生成的模型资产使用 [MIT 许可证](LICENSE)。依赖库分别遵循各自许可证，
不在本仓库内重新分发。未包含的数据与第三方论文不在此授权范围内。
使用时请引用论文；作者和标题列于 `CITATION.cff`，不填写尚不存在的 DOI 或仓库地址。

欢迎通过 GitHub issue 提交可复现问题，附环境、命令和模拟示例，不上传受限现场数据。
详见 `CONTRIBUTING.md`，上传步骤见 `docs/GITHUB_UPLOAD.zh-CN.md`。
