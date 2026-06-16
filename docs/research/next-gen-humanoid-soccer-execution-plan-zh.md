# 下一代人形机器人足球执行计划

## 目的

本文档把 `docs/research/next-gen-humanoid-soccer-roadmap.md` 中的高层路线图转化为可执行计划。

它用于回答四个问题：

1. 接下来应该构建什么系统。
2. 哪些接口应该分隔各层。
3. 当前仓库中应该先构建什么。
4. 未来 3 个月和 6 个月如何分阶段推进。

## 执行摘要

当前仓库是以下工作的强基础：

- 人形机器人 motion tracking
- 带位置泛化的足球踢球
- recurrent PPO 训练
- 滚动球处理
- 围绕首次接触和踢后球结果的 reward shaping

但它还不是以下工作的强基础：

- 通用 motor foundation modeling
- 连续战术决策
- 超越射门中心原语的多技能足球行为
- 多智能体 self-play
- 在线学习式 skill generation

因此推荐顺序是：

1. 审计并稳定当前 PAiD 风格流水线。
2. 将 Stage I 泛化为可复用的 whole-body sport foundation。
3. 用参数化技能控制替代 motion retrieval / 固定技能条件。
4. 加入战术层。
5. 加入对手，然后加入 self-play。

## 近期收敛场景：一次完整进攻

为了避免一开始就把问题扩展到完整足球智能，第一版目标应该收敛到一个具体、可训练、可部署验证的闭环：

```text
RoboJuDo 行走模型
    -> 走到球场 / 起始区域
    -> 切换到足球模型
    -> 基于当前球和目标观测接近足球
    -> 调整身体和步点
    -> 带球或直接射门
    -> 踢后恢复
    -> 切回 stand / locomotion 或保持足球状态
```

这个场景暂时不追求：

- 完整多智能体战术
- VLA 式端到端生成动作
- 完整 self-play league
- 无边界的通用运动大模型
- 从图像直接输出关节动作

它要先证明几件更核心的事情：

- RoboJuDo 中的 locomotion / soccer / stand 切换是稳定的。
- 足球 policy 能使用部署时真实可获得的观测，而不是依赖仿真专有信息。
- 球和目标的表达方式在训练、仿真部署、实机部署之间一致。
- 低层 policy 能完成接近、调整、触球、踢后恢复这一段连续行为。
- 高层以后只需要注入意图和目标，而不是直接参与每一步低层平衡。

## 系统架构

### Layer 0：Reflex Arbitration

职责：

- 高频监控稳定性、动作可执行性和切换安全性
- 在失稳、摔倒、触球失败或观测异常时抑制当前技能
- 触发 recover、stand、get-up 或重新接近

输入：

- base 姿态和角速度
- projected gravity
- contact / foot support summary
- ball visibility / confidence
- 当前 skill phase

输出：

- allow / inhibit / abort
- recover / stand / keep-running 等反射级状态
- 给 Motor Coordinator 的安全约束

注意：

- 它不是战术大脑。
- 它也不应该等高层决策模型低频判断后才工作。
- 它应该和运动基座同频或接近同频运行。

### Layer A：Motor Foundation

职责：

- 高频稳定的 whole-body control
- locomotion、recovery、接触时序和球交互原语

输入：

- 本体感知
- 可选的短时域 latent skill signal
- 可选的 motion-tracking targets

输出：

- policy frequency 下的低层动作命令

目标性质：

- 扰动下鲁棒
- 接触感知
- 可复用于足球和非足球任务

### Layer B：Soccer Skill Adapter

职责：

- 把通用身体智能转化为足球专项动作族

输入：

- motor-foundation latent state
- soccer task state
- skill parameters

输出：

- skill-conditioned latent 或 residual actions

目标性质：

- 保留 motor prior
- 优先用小型可训练头适配
- 支持 kick、trap、dribble、pass、recover、intercept

### Layer C：Tactical Decision Model

职责：

- 选择并参数化下一个行为 chunk

输入：

- ball state
- self state
- teammate states
- opponent states
- target geometry
- uncertainty and visibility

输出：

- skill type
- skill parameters
- hold / switch / abort signals
- short-horizon latent plan

目标性质：

- 闭环
- object-centric
- 在部分可观测下鲁棒

### Layer D：Match-Scale Learning

职责：

- 把孤立足球技能转化为竞赛行为

输入：

- 完整多智能体比赛状态
- tactical objective

输出：

- 技能和站位上的长时域决策分布

目标性质：

- self-play compatible
- opponent-aware
- 支持配合和压力适应

### 并联协同关系

整体控制结构应理解为并联协同，而不是串行调用：

```text
                    perception / world state
                             |
              --------------------------------
              |                              |
       Tactical Brain                  Reflex Monitor
       1-5 Hz                          50-200 Hz
              |                              |
              | Skill Intent                 | override / inhibit
              v                              v
              -------- Motor Coordinator -----
                             |
                      Motor Foundation
                      50-200 Hz
                             |
                        joint targets
```

这里的 `SkillCommand` 更准确地说是 intent / modulation signal，而不是“高层命令低层照做”的刚性指令。高层只决定目标、技能倾向和短时域意图；反射层随时可以抑制或覆盖；低层运动基座始终负责平衡、接触和执行。

## 接口规范

最重要的工程步骤是在加入新模型之前定义接口。

### Interface 1：`MotorState`

最小字段：

- base pose and velocity
- joint position and velocity
- projected gravity
- contact summaries
- previous action
- 可选 recurrent hidden state handle

可能来源：

- `tracking_env_cfg.py` 中当前 policy observation terms

### Interface 2：`SoccerState`

最小字段：

- local frame 下的球位置
- local frame 下的球速度
- local frame 下的 target / goal 位置
- strike-leg prior
- target destination
- confidence / noise proxy

可能来源：

- 当前 `target_point_pos`
- 当前 `target_destination_pos_local`
- 未来新增：显式 ball velocity observation

### Interface 3：`SkillCommand`

它应该成为战术层和技能层之间的核心契约。

最小字段：

- `skill_type`：run、approach、trap、dribble、pass、shoot、recover
- `kick_leg`
- `approach_heading`
- `contact_time_hint`
- `contact_point_hint`
- `ball_velocity_target`
- `post_contact_heading`
- `recovery_style`

语义：

- 它是意图和调制信号，不是逐帧动作脚本。
- 它允许高层指定“想做什么”，但低层仍然决定“怎么稳定地做”。
- 它应该允许 Reflex Arbitration 在不改变高层目标的情况下临时 hold、abort 或 recover。

短期实现：

- 表示为追加到 observation 的结构化 tensor block

长期实现：

- 显式 dataclass / manager term / command term

### Interface 4：`ReflexState`

最小字段：

- `is_falling`
- `is_recovering`
- `support_quality`
- `skill_allowed`
- `abort_requested`
- `recover_mode`

目的：

- 把安全、恢复和动作抑制从战术层中拆出来。
- 避免高层模型低频地处理所有失稳细节。
- 让实机部署中 stand / locomotion / soccer 之间的切换更可控。

### Interface 5：`LatentSkillChunk`

只在加入 diffusion / flow / chunked planner 时使用。

最小字段：

- latent token sequence 或 vector
- validity horizon
- 可选 contact schedule

目的：

- 保持学习式高层生成与低层动作执行分离

## 当前仓库映射

以下文件是当前代码库中最相关的执行表面。

### 训练入口

- `scripts/rsl_rl/train_multi.py`
- `scripts/rsl_rl/play_multi.py`

这些是应该保留为外层 orchestration entrypoints 的正确位置。

### 当前 Stage I / Stage II 环境拆分

- `source/whole_body_tracking/soccer/tasks/tracking/config/g1/__init__.py`
- `source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py`

当前映射：

- terrain motion tracking stage：
  `Tracking-Terrain-G1-RNN-v0`
- flat kick generalization stage：
  `Tracking-Flat-G1-SoccerDestination-RNN-v0`

### 观测和奖励定义

- `source/whole_body_tracking/soccer/tasks/tracking/tracking_env_cfg.py`
- `source/whole_body_tracking/soccer/tasks/tracking/mdp/observations.py`
- `source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards.py`
- `source/whole_body_tracking/soccer/tasks/tracking/mdp/terminations.py`

这些文件已经编码了论文 Stage I 和 Stage II 的大部分逻辑。

### Motion / Command 逻辑

- `source/whole_body_tracking/soccer/tasks/tracking/mdp/commands_multi_motion.py`
- `source/whole_body_tracking/soccer/tasks/tracking/mdp/commands_multi_motion_soccer.py`

这些是高优先级文件，因为它们包含：

- multi-motion handling
- adaptive sampling
- 足球摆放
- target destination generation
- rolling-ball initialization

### Distillation Hook

- `source/whole_body_tracking/soccer/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `source/whole_body_tracking/soccer/tasks/tracking/config/g1/__init__.py`

student-teacher config 已经存在。这很可能是试验 foundation-model-to-soccer adaptation pipeline 的最近、最干净入口。

## 近期工作包

### Work Package 0：One-Attack Task Spec

目标：

- 把第一版足球系统收敛成一次完整进攻，而不是完整比赛智能

任务定义：

- RoboJuDo locomotion 走到起始区域
- 切换到 soccer policy
- 使用当前 ball / goal / body observation 闭环接近足球
- 完成一次带球、调整或直接射门
- 踢后恢复并允许切回 locomotion / stand

交付物：

- 训练场景定义
- 仿真评估脚本
- 部署观测契约
- 切换和恢复条件
- 成功 / 失败指标

成功标准：

- 同一个任务能在 HumanoidSoccer 训练环境、RoboJuDo 仿真部署、RoboJuDo real 部署中用同一套 observation 语义描述。
- 不依赖多智能体、不依赖完整球门识别、不依赖 VLA。

### Work Package 1：Paper-To-Code Audit

目标：

- 在论文论述和仓库实现之间建立精确映射

交付物：

- 哪些论文组件被完整实现
- 哪些只是近似实现
- 哪些缺失

优先文件：

- `soccer_flat_env_cfg.py`
- `commands_multi_motion_soccer.py`
- `rewards.py`
- deployment / exporter utilities

成功标准：

- 一份从论文 section 到 file/function/config 的可追溯矩阵

### Work Package 2：General Motor Foundation Gap Analysis

目标：

- 判断当前 Stage I 相比真正 sport foundation 有多窄

审计问题：

- 当前包含哪些 motions
- 是否存在 get-up / recovery / turn / sidestep
- 是否有显式 dribble / trap / pass 数据
- motion distribution 是否过于偏向踢球

交付物：

- motion taxonomy
- missing-skill inventory
- proposed dataset expansion list

### Work Package 3：Skill Interface Refactor

目标：

- 在改变模型类别之前引入显式 `SkillCommand` 接口

短期代码方向：

- 在 motion command 中加入结构化 skill-conditioning fields
- 在 observations 中暴露这些字段
- 允许固定和 scripted overrides，便于调试

交付物：

- 一个显式 skill-conditioning contract
- 一个支持手动技能控制的 debug environment

为什么现在做：

- 这能降低后续 tactical-model 工作的风险

### Work Package 4：Rolling-Ball and Ball-State Upgrade

目标：

- 强化当前 soccer state 表达

可能改动：

- 显式 local-frame ball velocity observation
- ball confidence / visibility indicator
- 短时域 filtered state，用于部署一致的噪声注入

交付物：

- 升级后的 `SoccerState`
- 消融：显式速度是否比仅依赖 LSTM 隐式记忆更有帮助

### Work Package 5：Tactical Layer Prototype

目标：

- 构建第一个决策模型，但暂时不要求多智能体比赛

建议 version 1：

- 基于结构化状态的 object-centric MLP / transformer
- 输出 `SkillCommand`
- 初始在 1v0 delayed-shot / reposition 场景中训练

暂时不做：

- 不做大型 VLA stack
- 不做 raw-image end-to-end policy

原因：

- 先证明接口和控制分解有效

### Work Package 6：Generative Skill Planner Prototype

目标：

- 评估 diffusion 或 flow-based 中层生成是否真的有帮助

推荐顺序：

1. 非生成式 skill-parameter model baseline
2. chunked autoregressive baseline
3. flow-matching latent skill model
4. 只有在多样性明确重要且延迟可接受时再用 diffusion

关键指标：

- 在球运动变化和扰动下，生成式规划是否优于参数回归

### Work Package 7：Self-Play Soccer Curriculum

目标：

- 从孤立射手控制转向足球行为

推荐顺序：

1. striker vs empty goal
2. striker vs goalie
3. striker vs defender
4. 2v2 pass-and-shoot
5. small-sided self-play

关键基础设施：

- opponent policy pool
- 先用 scripted opponents
- 超越 raw reward 的指标

## 三个月计划

### Month 1

重点：

- 定义并跑通一次完整进攻的最小闭环

任务：

- 写清楚 one-attack task spec：起始状态、球和目标分布、切换条件、成功标准
- 对齐 HumanoidSoccer 训练环境和 RoboJuDo 部署中的 `SoccerState`
- 验证 soccer policy 的 ball / goal observation 不依赖仿真专有数据
- 验证 RoboJuDo 中 locomotion -> soccer -> stand / locomotion 的切换
- 保留 paper-to-code audit，但只优先审计会影响一次进攻闭环的部分
- 完成 paper-to-code traceability document
- 验证 adaptive sampling 实现
- 验证 rolling-ball 实现
- 验证 sim-to-real 组件哪些存在、哪些缺失

输出：

- 一份 one-attack task spec
- 一份训练 / 仿真部署 / 实机部署 observation 对齐表
- 一份切换稳定性报告
- 一份围绕一次进攻的最小 audit report

### Month 2

重点：

- 为一次完整进攻准备接口和训练

任务：

- 引入 `SkillCommand`
- 引入最小 `ReflexState`
- 重构 soccer command pipeline，使其接受结构化 skill parameters
- 加入显式 ball velocity observation
- 构建 scripted/manual one-attack tests
- 用现有 soccer policy 作为第一版低层技能，先不引入复杂 tactical model

输出：

- 第一个稳定 interface layer
- 第一批 one-attack 调试工具
- 第一个可以评估接近、触球、踢后恢复的仿真任务

### Month 3

重点：

- 1v0 一次完整进攻中的 tactical layer v1

任务：

- 在结构化状态上实现一个小型 tactical model
- 与当前 motion-selection logic 对比
- 增加“直接射门次优、需要 reposition / dribble”的任务变体
- 评估它是否真正提升一次进攻成功率，而不是只提升单次射门

输出：

- 第一批证据：高层意图注入是否能改善一次完整进攻闭环

## 六个月计划

### Months 4-5

重点：

- foundation 和 generative planning 探索

任务：

- 扩展 kicks 之外的 motion categories
- 加入 recovery 和 turning 数据
- 在更宽低层 policy 上原型化 adapter-based football specialization
- benchmark regression vs flow-based skill generation

输出：

- 更宽的 motor prior
- 第一个中层 planning benchmark

### Month 6

重点：

- 交互式足球

任务：

- 加入 goalkeeper 或 defender
- 定义多智能体指标
- 从窄场景开始 self-play curriculum

输出：

- 第一次从孤立足球技能转向对抗足球行为

## 指标

未来工作不要只用射门成功率评估。

### 低层 / 技能指标

- motion tracking quality
- recovery success
- contact correctness
- kick direction error
- kick speed
- post-contact stability

### 决策指标

- shot timing quality
- delayed strike 条件下的成功率
- 能否选择 reposition，而不是立即执行差射门
- moving-ball 场景下的 time-to-shot
- re-touch / second-touch success

### 比赛指标

- possession retention
- expected shot quality proxy
- 小场 self-play 胜率
- 对手压力下的鲁棒性

### 真实世界指标

- deployment latency
- perception dropout robustness
- ball-parameter mismatch 下的行为
- indoor/outdoor consistency

## 模型建议

### Tactical Layer

优先使用：

- structured-state transformer 或 object-centric network

不要优先使用：

- full image-language-action foundation model

原因：

- 当前仓库和问题阶段还不值得引入完整 VLA 复杂度

### Mid-Layer Generator

优先使用：

- 如果在线延迟重要，优先 flow matching

后续使用：

- 只有当多模态和 expressive motion chunk generation 明确值得更慢采样时，才使用 diffusion

### Low-Level Controller

保留：

- recurrent、稳定、高频 motor policy

不要替换为：

- 控制频率下的 raw generative action model

## 工程原则

- 保持小而可审查的 diff
- 先加接口，再加大模型
- 在大范围重构前，用当前 baseline 衡量新层收益
- 将 sim-only 想法与 deployment-ready code paths 分开
- 除非 baseline 证明值得，否则不要把 tactical planning 和低层 contact stabilization 混到一个模型里

## 当前仓库中的建议下一步任务

推荐顺序：

1. 写出 one-attack task spec：从 RoboJuDo 行走切入足球模型，到带球 / 调整 / 射门 / 恢复。
2. 将训练、HumanoidSoccer play、RoboJuDo sim、RoboJuDo real 的 observation contract 抽取成同一份 spec。
3. 审计 `commands_multi_motion_soccer.py` 中 ball / goal / motion command 的生成方式，只看会影响 one-attack 的部分。
4. 设计并实现 `SkillCommand v1` 和最小 `ReflexState`。
5. 构建 one-attack sim task，先用现有 soccer policy 做 baseline。
6. 再加入小型 tactical-layer sandbox，让机器人在直接射门、一步调整、短带球之间做选择。

## 当前阶段的完成定义

当前探索阶段只有在以下条件满足时才算完成：

- 一次完整进攻任务已经被明确定义
- 训练、仿真部署、实机部署的足球观测语义已经对齐
- RoboJuDo 中 locomotion / soccer / stand 切换稳定可测
- 第一个显式跨层接口已经存在
- 一个 tactical prototype 能够通过该接口改善 one-attack baseline

到那时，项目才算从“论文讨论”进入“下一代系统构建”。
