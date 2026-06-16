# 人形机器人踢球蒸馏到实机的改进建议

本文整理当前 G1/Saya 踢球任务在 teacher-student distillation 阶段，为了更好 sim-to-real 可以优先改进的方向。

## 当前判断

当前蒸馏链路的主问题不在网络结构，而在于 student 训练条件是否足够接近实机部署条件。

已经确认过的关键点：

- Teacher checkpoint 使用了 160 维 actor observation normalizer，蒸馏时必须把该 normalizer 接到 teacher label 和 student 初始化路径上。
- Warm-start 不等于普通 resume。它只是用 teacher 的 `memory_a + actor + obs normalizer` 初始化 student，后续训练仍然是 student 模仿 teacher action。
- 蒸馏 loss 降低不一定代表 rollout 稳定。人形踢球对相位、落脚点、球相对位置和接触时机非常敏感，小 action error 也可能导致摔倒。

## 优先级最高的改进

### 1. Student observation 尽量等于实机可获得 observation

Student 不应该依赖仿真里干净、全局、完美的球和目标信息。它应尽量使用实机部署时可获得的信息：

```text
机器人本体状态：
  projected_gravity
  base / torso angular velocity
  joint_pos
  joint_vel
  last_action

视觉 / 感知状态：
  ball position in torso/camera/base frame
  ball visibility flag
  ball confidence
  time_since_last_seen
  optional ball velocity estimate
  goal/vector cue
```

如果实机通过相机 YOLO + depth 得到球位置，那么仿真蒸馏里的 student obs 也应该模拟这种观测，而不是长期依赖 perfect state。

### 2. 蒸馏时加入感知噪声、丢帧和延迟

踢球任务对球位置误差极其敏感。训练时如果一直使用干净球坐标，实机很容易出现提前碰球、踢空、最后一脚时机错的问题。

建议先从简单配置开始：

```text
ball_pos += Normal(0, [0.03, 0.03, 0.02])
10% frames dropout
perception update every 3~5 policy steps
dropout 时 hold last visible ball position
obs 中加入 ball_valid / time_since_last_seen
```

后续可以再加入：

```text
偶发 outlier
深度方向更大的噪声
检测置信度扰动
相机外参轻微扰动
```

### 3. 加入 action delay 和执行器随机化

实机不会像仿真一样动作立刻、完美作用到关节。部署时存在 inference delay、通信延迟、电机响应延迟和 PD 误差。

建议蒸馏或后续 student fine-tune 中加入：

```text
action delay: random 1~3 control steps
action low-pass filtering
PD gain randomization
motor strength randomization
joint friction / damping randomization
latency jitter
```

这对踢球比普通行走更重要，因为踢球动作冲击大，接触窗口短。

### 4. 加强球和接触物理 domain randomization

踢球 sim-to-real 的核心难点之一是脚-球-地面的接触。

建议随机：

```text
ball mass
ball radius
ball friction
ball restitution
ground friction
foot friction
contact restitution
ball initial position
```

球初始位置不要完全固定。即使当前只验证单 motion，也建议加入小扰动：

```text
ball_pos x/y: +/- 3~8 cm
```

这样 student 不会只学会踢一个精确点。

## 训练方法上的改进

### 5. 保持 student rollout 的 DAgger 思路

当前 `DistillationRunner` 已经接近 DAgger 的核心形式：

```text
student rollout 访问状态
teacher 在这些状态上给 action label
student 用 behavior cloning loss 学 teacher
```

后续可以进一步增强为更标准的 DAgger：

```text
保留历史 rollout 数据
构建 aggregated dataset / replay buffer
新旧数据混合训练
对失败边缘状态提高采样比例
```

这样 student 不会只依赖最近一个 rollout batch，能覆盖更多偏离 teacher trajectory 的状态。

### 6. 蒸馏后做 student-only PPO fine-tune

纯蒸馏通常只能让 student “像 teacher”，但不保证它在 deploy obs 下 reward 最优。

建议最终训练路线是：

```text
1. Teacher PPO:
   privileged / clean obs，先学会稳定踢球。

2. Student distillation:
   deploy obs，student rollout，teacher action label。

3. Student PPO fine-tune:
   仍然使用 deploy obs，小学习率，强 domain randomization。

4. Export / sim-to-real:
   使用和部署一致的 observation、normalizer、action delay 和滤波设置。
```

第 3 步很关键。它能让 student 在自己的观测条件和自己的 rollout 分布下恢复闭环稳定性。

### 7. 可以混合 imitation loss 和 RL loss

如果纯 behavior cloning 后 reward 上不去，可以考虑：

```text
loss = imitation_loss + small PPO/RL loss
```

或者阶段式：

```text
先 distill 到能稳定站住和完成动作
再用 student obs 接 PPO fine-tune
```

这样可以避免 student 只追 teacher action MSE，而忽略别摔、接触时机、球速度方向等最终任务指标。

### 8. 给 student 显式 phase / time cue

踢球是强时序任务。RNN 可以自己记 phase，但感知丢帧、动作延迟、摔倒边缘状态会让 phase 估计变难。

可以考虑给 student 加：

```text
motion phase
time step / normalized time
remaining time
```

或者额外蒸馏：

```text
student hidden state vs teacher hidden state
```

这不是第一优先级，但在动作相位不稳时值得尝试。

## 建议实验顺序

优先不要一次改太多。建议按下面顺序推进：

```text
A. 当前 normalizer + warm-start distillation 跑通
B. 加 ball obs noise/dropout/delay
C. 加 action delay / PD randomization
D. 加球和接触物理随机化
E. 用 student checkpoint 做 PPO fine-tune
F. 再做 no-warm-start ablation
```

其中 A 是当前主线。F 只是对照实验，不应该作为主路线。

## 建议 ablation

为了确认每个因素的作用，可以跑以下对照：

```text
1. warm-start + normalizer
2. no warm-start + normalizer
3. warm-start + perception noise/dropout
4. warm-start + action delay
5. distillation only
6. distillation + student PPO fine-tune
```

判断指标不要只看 `Loss/behavior`，还要看：

```text
Mean reward
Mean episode length
play 时是否能站稳
踢球时机是否正确
球速度方向是否对
对球初始位置扰动是否鲁棒
```

## 结论

当前最重要的方向是让 student 的训练观测、延迟、噪声、执行器和接触条件尽量接近实机。  

更换网络结构不是第一优先级。先把以下链路做扎实：

```text
teacher PPO -> normalized warm-start distillation -> noisy deploy obs distillation -> student PPO fine-tune -> export
```

这样比单纯追求更低 behavior loss 更有利于 sim-to-real。

## 真机数据采集与微调

真机数据有必要采集，但建议放在 sim 中 student 已经基本稳定之后，而不是在 sim 里还一动就摔时直接上真机数据微调。

对踢球任务来说，真机数据最有价值的用途不是重新学习踢球动作，而是校正 sim-to-real gap：

```text
实际关节响应延迟
PD / 电机动力学误差
相机球位置噪声和延迟
脚-球接触后的真实球运动
机器人起脚 / 支撑脚的真实姿态偏差
```

如果 student 在 sim 里还不稳定，真机数据微调意义不大，因为主要问题还在训练链路、观测定义、normalizer 或蒸馏设置。  

如果 student 在 sim 里已经能稳定完成动作，但实机上出现踢偏、提前碰球、力度不对、最后一步不稳等问题，真机数据就很有价值。

### 建议采集的数据

建议至少采三类数据：

```text
1. 实机 observation / action log
   policy obs
   action
   joint state
   IMU
   torso quat
   timestamp

2. 感知数据
   camera / depth 推出来的 ball pos
   ball_valid
   confidence
   perception timestamp
   estimated delay

3. 结果数据
   是否踢到球
   球初速度 / 方向
   机器人是否摔倒
   接触时刻
   失败原因标注
```

这些数据最好全部带统一时间戳，否则很难判断到底是感知延迟、控制延迟还是动力学误差导致失败。

### 推荐使用方式

不建议一开始就用真机数据端到端微调整个 RNN policy。更稳妥的顺序是：

```text
A. System identification
   用真机 log 反推 sim 中的 delay、PD、friction、ball 参数。
   然后回 sim 里用校准后的参数继续训练。

B. Behavior cloning 校正
   只用真机上稳定片段，小学习率 supervised fine-tune。
   目标是微调动作细节，而不是重新学习完整踢球策略。

C. Residual / adapter fine-tune
   冻住主网络，只训练小 adapter 或 action residual。
   这样比直接改完整 RNN policy 风险低。
```

直接用少量真机数据端到端微调整个 policy 风险较高：

```text
数据量通常不够
失败数据分布偏
容易破坏 sim 中已经学到的稳定性
RNN hidden dynamics 容易被小数据带偏
```

### 建议整体路线

更推荐的 sim-to-real 闭环是：

```text
1. 先把当前 distillation 在 sim 中跑到稳定
2. 加 perception noise / delay / motor delay / contact randomization
3. 实机低风险试跑并采 log
4. 用真机 log 做 system identification
5. 回 sim 中用校准后的参数继续 student fine-tune
6. 最后再考虑少量 BC / residual 微调
```

结论：

```text
真机数据有必要采；
但主要用于校准和小幅适配；
不建议作为当前主训练手段；
更不建议在 sim student 还不稳定时直接端到端微调整个策略。
```
