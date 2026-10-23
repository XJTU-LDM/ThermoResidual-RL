# ThermoResidual-RL

基于物理先验的残差强化学习地热发电与制冷协同控制。

[English](README.md) · [实验协议](docs/PROTOCOL.md) · [数据与单位](docs/DATA.md)

本仓库只保留论文最终使用的程序、冻结配置和模型权重，不包含历史实验目录、
旧版实现、现场原始记录或保密项目报告。代码采用 [MIT 许可证](LICENSE)。

## 算法概览

ThermoResidual-RL 以名义热力学分配作为动作参考，根据已到达的运行测量历史学习
动态补偿。残差动作映射覆盖完整支路流量范围，并与全动作软演员—评论家方法
（soft actor-critic，SAC）使用相同的流量和变化率约束。

[![ThermoResidual-RL 控制流程](docs/figures/control_framework.png)](docs/figures/control_framework.png)

测量进入共享历史观测与名义分配器，残差网络产生流量修正，投影后的命令驱动
源—负荷耦合模型；新的闭环响应形成下一时刻的观测和奖励。压力集成网络用于预测
与缺测替代。图中底部的小曲线为示意，不是实验结果。

守恒随机参考模型包括地层热记忆、氨库存与内能、液压输运、发电、吸附制冷、
冷凝排热和回液。扰动作用于模型参数，压力和取热由模型状态及控制动作共同决定。

| 命令名称 | 控制器 |
|---|---|
| `rrl` | ThermoResidual-RL |
| `sac` | 信息对齐的全动作 SAC |
| `pi` | 名义分配加比例积分反馈 |
| `gspid` | 增益调度比例积分微分控制 |
| `mpc` | 约束模型预测控制 |
| `physics` | 名义热力学分配 |

## 安装与快速运行

已测试 Python 3.11。建议使用独立环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python reproduce.py --verify-only
python -m pytest -q
python reproduce.py --suite demo --workers 2 --output outputs/demo
python plot_results.py --input outputs/demo/demo_256 --output outputs/demo/figures
python summarize_results.py --input outputs/demo/demo_256/metrics.csv --output outputs/demo/summary.csv
```

Windows 激活命令为 `.venv\Scripts\activate`。普通评价支持 CPU；CUDA 训练需要
与本机驱动兼容的 PyTorch。即使神经网络使用 GPU，物理求解部分仍主要消耗 CPU。
上述演示为 256 步，前 200 步为预热，用于检查程序运行，不代替正式精度评价。

## 正式评价

```bash
python reproduce.py --suite main --workers 4 --output outputs/evaluation
python reproduce.py --suite stress --workers 4 --output outputs/evaluation
python reproduce.py --suite weights --workers 4 --output outputs/evaluation
```

| 评价配置 | 轨迹数 | 内容 |
|---|---:|---|
| `main` | 480 | 六种控制器、四类目标、四工况、五个过程随机实现 |
| `stress` | 840 | 五种控制器、参考工况及六种对象参数干预 |
| `weights` | 320 | 两组额外跟踪权重下的四个冻结学习策略 |
| `sensor` | 792 | 60 s 压力采集、10 s 控制下的压力接口诊断 |

通过 `--controllers`、`--profile`、`--condition` 和 `--limit` 可以运行子集。
`sensor` 是独立于正常 10 s 采集设置的跨采样接口诊断，可用
`python reproduce.py --suite sensor --workers 4` 运行。它保留当前绝压设置下的
全部诊断条件，不将跨采样诊断等同于已证明跨采样鲁棒性。全部轨迹均为模拟数据，
过程随机实现反映冻结策略对扰动的响应变化，不是多次独立训练。

## 训练与传统控制器整定

RRL 和 SAC 采用相同的双层 256 单元网络、归一化、初始网络权重、空经验回放、
SAC 超参数和每次 768,000 次交互预算。按冻结开发窗口的共同代价选择检查点。

```bash
# 短运行检查
python train.py --controller rrl --device cpu --smoke --output outputs/rrl_smoke
python train.py --controller sac --device cpu --smoke --output outputs/sac_smoke
python tune_baselines.py --controller pi --smoke --output outputs/pi_smoke

# 完整训练：在两个终端分别启动，可使用两块 GPU
python train.py --controller rrl --device cuda:0 --output outputs/rrl
python train.py --controller sac --device cuda:1 --output outputs/sac

# 权重敏感性，另支持 --weight-ratio 2
python train.py --controller rrl --weight-ratio 0.5 --device cuda:0 --output outputs/rrl_weight_0p5

# 仅在开发窗口整定，评价时冻结参数
python tune_baselines.py --controller pi --output outputs/tune_pi
python tune_baselines.py --controller mpc --output outputs/tune_mpc
python tune_baselines.py --controller gspid --output outputs/tune_gspid
```

训练必须使用新的输出目录，从包内空学习器状态开始，不从已训练策略继续训练。
PI、MPC 和 GS-PID 分别搜索 24、9、54 组参数。观测、动作映射、奖励函数和选模
规则详见 [PROTOCOL.md](docs/PROTOCOL.md)。

## 目录与数据

```text
thermoresidual/       最终物理模型、控制、学习和评价程序
assets/model/        冻结物理参数、过程扰动及积分设置
assets/policies/     主比较和当前权重敏感性策略
assets/pressure/     五成员因果压力预测器
configs/             冻结评价任务
data/synthetic/      标明为模拟数据的示例及单位说明
docs/                协议、复现说明和一幅算法图
tests/               数值、接口、资产和隐私检查
tools/               发布前检查工具
```

可以生成新的合成示例并检查发布目录：

```bash
python generate_synthetic_data.py --output outputs/synthetic
python tools/audit_public_release.py
```

包内不包含现场 CSV/Excel、原始测量日志、项目报告、账号信息或非空经验回放。
提供的模型权重及有效系数属于派生模型资产。压力网络可直接推理；原始拟合依赖
另行授权的数据，本包不分发这些记录。[DATA.md](docs/DATA.md) 说明数据来源与单位，
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) 说明运行、输出和验证方式。
