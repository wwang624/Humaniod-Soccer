# 下一代人形机器人足球路线图

## 背景

本文档记录了对论文 `Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework` 的复盘结论，以及后续关于其局限性和竞赛级人形机器人足球下一步研究方向的讨论。

目标是在未来的 Codex/OMX 会话之间保留上下文，让工作能够从同一个概念基线继续，而不是每次重新推导整套系统。

## 论文总结

PAiD 很好地解决了一个窄但有价值的问题：

- Stage I 通过运动跟踪学习稳定、类人的踢球原语。
- Stage II 加入轻量感知和任务奖励，实现对球位置的泛化以及对滚动球的处理。
- Sim-to-real 重点放在接触参数对齐和结构化观测噪声上。

这篇论文作为 `技能获取与迁移` 工作很强；但作为真正的 `足球智能` 工作则相对较弱。

## 主要批评

核心弱点在高层决策层。

PAiD 学到的主要是一个强射门技能流水线，而不是一个具备竞赛能力的足球智能体。系统仍然接近于：

- 选择或条件化到某个 motion
- 接近足球
- 执行踢球

它还没有真正解决：

- 何时射门、何时延迟、何时重新调整位置
- 在压力、扰动或对手运动下如何连续自适应
- 如何执行停球、带球、恢复、二次触球等多触球行为
- 如何对队友、防守者、空间和战术价值进行推理
- 如何把技能执行变成赢得比赛的行为

简而言之：

- PAiD 是一个高质量的 `射门技能系统`
- 但还不是完整的 `足球决策系统`

## 下一代工作的核心论点

下一套系统不应该只是“更好的踢球策略”。

它应该是一个显式分层的 `足球智能体栈`：

1. 通用运动基础
2. 足球技能适配
3. 战术决策模型
4. 多智能体 self-play 与真实世界适配

## 建议架构

### 并联大小脑协同架构

下一代系统不应该被理解成一个串行的 VLA 或生成式流水线：

```text
vision / language / state -> planner -> action sequence -> robot
```

更合理的形式是并联的分布式控制：

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

这个结构的核心不是“高层网络生成低层动作”，而是：

- Tactical Brain 低频运行，负责战术意图：什么时候推进、往哪里带、什么时候射门、是否需要等待或重新调整。
- Reflex Monitor 高频运行，负责安全和即时响应：失稳、碰撞、摔倒、球突然偏离、动作不可执行时的抑制或覆盖。
- Motor Coordinator 把战术意图、反射状态和当前技能状态融合成低层 policy 可理解的调制信号。
- Motor Foundation 高频闭环执行，负责真正的 whole-body balance、接触时序、步态和球交互。

可以简化成一句话：

```text
Brain modulates, Reflex arbitrates, Motor executes.
```

因此，低层运动基座既不应该只是一个固定踢球 policy，也不应该一开始就追求完全通用、无边界的 Sonic 式全能模型。更现实的第一版目标是：

- 以 whole-body sport foundation 为方向
- 先覆盖足球中最核心的移动、调整、触球、恢复
- 通过足球专项 adapter / skill head 承载射门、带球、停球等技能
- 给高层留下 `intent / skill / target` 注入接口，而不是让高层直接控制关节

### 第 1 层：通用运动基础

Stage I 应该从“踢球跟踪”扩展为通用的人形机器人运动基础模型。

需要覆盖：

- 移动：走、跑、加速、减速、切向、侧移、转身
- 稳定：推搡恢复、踉跄恢复、摔倒恢复、起身
- 接触组织：支撑腿落点、摆腿时序、平衡修复
- 球交互原语：停球、轻拨、推球、带球、传球、射门

数据来源应该混合：

- motion tracking / mocap
- 遥操作或人类示范
- 扰动 curriculum
- 恢复 curriculum
- 足球专项的富接触 motion 数据

重要原则：

- 这不只是 locomotion foundation。
- 它应该是 whole-body sport foundation。

### 第 2 层：足球技能适配

在运动基础之上适配足球。

不要把它当成固定踢法库，而应该学习一个参数化技能空间，变量包括：

- 技能类型
- 踢球腿
- 接近角度
- 接触时机
- 接触点
- 期望球方向
- 期望球速度
- 触球后的恢复朝向

推荐的适配策略：

- 冻结大部分 motor backbone
- 使用 residual heads / adapters / LoRA 风格的专项化模块
- 通过 RL fine-tuning 做足球专项执行优化

避免从零开始做完整端到端重训，因为这有损坏通用身体控制先验的风险。

### 第 3 层：战术决策模型

这是目前最重要的缺失部分。

战术层应该运行在结构化比赛状态上，而不是只看球和球门几何。输入应该包括：

- 球状态
- 自身状态
- 队友状态
- 对手状态
- 空间 / 通道几何
- 不确定性 / 可见性状态

输出不应该是原始关节动作，而应该是：

- latent skill chunks
- 技能参数
- 短时域意图变量
- switch / hold / abort 决策

这一层应该回答：

- 我现在是否应该射门
- 是否应该再控一脚
- 应该用哪只脚
- 最优出球速度是什么
- 触球后身体应该如何面向下一状态

### 第 4 层：多智能体 Self-Play

要具备竞赛能力，系统必须从 1v0 技能执行转向交互式足球。

推荐 curriculum：

1. 1v0 技能执行
2. 1v0 加守门员
3. 1v1 进攻者对防守者
4. 2v2 传球和射门
5. 小场 self-play

重要 self-play 特性：

- 历史对手池
- population-based training
- 对手风格多样性
- 扰动和部分可观测性

没有这一层，系统会保持“技能强”，但不会具备稳定的战术鲁棒性。

## VLA、Diffusion 和 Flow Matching 的位置

### VLA

项目应该吸收 VLA 的思路，但不一定要做字面意义上的 vision-language-to-joint-action 流水线。

适合用途：

- 高层任务条件
- 战术或角色的语义 prompt
- 自我视觉、物体状态和比赛上下文之间的多模态融合
- 泛化的高层决策

不适合用途：

- 在人形机器人控制频率下直接做低层 whole-body control

结论：

- VLA 应该放在战术或高层规划层
- 不应该放在最终 motor-control loop

### Diffusion

Diffusion 吸引人的地方在于：

- 多模态技能生成
- motion prior 建模
- latent plan 生成
- 富接触短时域行为合成

但在高频下直接 diffusion-to-action 很可能太慢，也太脆弱，不适合作为最终低层控制器。

更好的用法：

- 生成 latent motion chunks
- 生成参数化 skill plans
- 引导一个更低层的稳定控制器

### Flow Matching

Flow matching 特别有潜力，因为它可能保留 diffusion 的部分表达能力，同时提升推理速度。

它是以下方向的强候选：

- 在线 skill-chunk 生成
- 短时域 action-plan 生成
- 在延迟敏感时替代 diffusion

实践结论：

- diffusion 或 flow 应该放在中间层
- 面向在线部署的生成优先考虑 flow matching
- 最终低层控制器保持确定性、稳定和高频

## 最重要的研究原则

不要做“更大的踢球模型”。

要做一个分层足球智能体：

- 高层：战术推理
- 中层：latent skill planning / chunk generation
- 低层：通用人形机器人运动基础

但第一版也不能直接把问题扩展到完整比赛智能。近期应该先收敛到一个窄而完整的任务闭环：

```text
RoboJuDo locomotion -> switch soccer policy -> approach / dribble / shoot -> recover / switch back
```

这个任务足够小，可以落地训练和实机验证；又足够完整，能暴露足球模型真正需要解决的问题：切换稳定性、球和目标观测、接近与调整、触球质量、踢后恢复，以及低层技能和高层意图之间的接口。

这个分离很可能是同时走向以下目标的最清晰路径：

- 更真实的足球行为
- 更高的比赛竞争力

## 建议研究路线

### Phase A：保留并扩展当前仓库

使用当前仓库作为执行基础，用于：

- paper-to-code mapping
- reward 和 observation 分析
- motion-skill pipeline 检查
- stage separation 检查

代码里需要立即回答的问题：

- 两个训练阶段在哪里分离
- motion tracking 如何体现在观测和奖励中
- adaptive sampling 是否如论文所述完整实现
- rolling-ball 支持是如何实现的
- physics-aware sim-to-real 部分是否在代码中完整存在

### Phase B：构建通用运动基础

把 Stage I 扩展成可复用的 motor prior。

交付物：

- 统一的运动技能 taxonomy
- 更广的 motion dataset
- recovery 和 disturbance curriculum
- 可复用的低层 policy API

### Phase C：用参数化技能规划替代 Motion Retrieval

用学习到的接口替代手工或 nearest-motion 选择：

- latent skill representation
- 参数化 kick / dribble / trap actions
- 可选的 diffusion/flow planner，用于短时域 skill chunk 生成

### Phase D：加入战术决策

构建一个高层决策层，对以下内容进行推理：

- 几何
- 不确定性
- 对手
- 队友
- 射门价值 vs 控球价值

### Phase E：转向 Self-Play

从射门技能走向足球行为。

优先关注的指标：

- 控球保持
- 压力下射门质量
- 踢球失败后的恢复
- 二次触球能力
- 小场比赛胜率

## 实际研究风险

主要技术风险：

- 过宽的 foundation model 可能稀释足球相关的接触技能
- 直接 VLA-to-action 可能太慢且不稳定
- diffusion 放错层级会导致在线不可用
- 全量 fine-tuning 可能破坏强 motor prior
- 没有 league/population 结构的 self-play 可能坍缩到脆弱行为

## 工作结论

当前最佳判断：

- 论文最强的部分是 progressive decomposition
- 论文最弱的部分是高层决策
- 下一次重大跃迁需要用分层决策 + 生成 + 鲁棒运动基础，替代以技能检索/执行为主导的系统

目标系统应该是：

- 不只是一个更好的射手
- 而是一个具备可复用身体智能的真实足球智能体

## 建议下一轮讨论

恢复这项工作时，可以继续以下方向之一：

1. 将论文的 Stage I 和 Stage II 论述映射到具体仓库代码。
2. 审计 adaptive sampling 和 sim-to-real 组件是否按论文描述实现。
3. 起草正式的下一代架构图和模块接口规范。
4. 把上面的路线图转成分阶段的工程 / 研究执行计划。
