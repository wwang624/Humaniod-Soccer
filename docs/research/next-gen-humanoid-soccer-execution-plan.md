# Next-Gen Humanoid Soccer Execution Plan

## Purpose

This document turns the high-level roadmap in
`docs/research/next-gen-humanoid-soccer-roadmap.md` into an executable plan.

It is intended to answer four questions:

1. What system should be built next.
2. Which interfaces should separate the layers.
3. What should be built first in the current repository.
4. How to stage the work over the next 3 and 6 months.

## Executive Summary

The current repository is a strong base for:

- humanoid motion tracking
- soccer kicking with positional generalization
- recurrent PPO training
- rolling-ball handling
- reward shaping around first contact and post-kick ball outcome

It is not yet a strong base for:

- general motor foundation modeling
- continuous tactical decision making
- multi-skill soccer behavior beyond shoot-centric primitives
- multi-agent self-play
- learned online skill generation

Therefore the recommended sequence is:

1. Audit and stabilize the current PAiD-style pipeline.
2. Generalize Stage I into a reusable whole-body sport foundation.
3. Replace motion retrieval / fixed skill conditioning with parameterized skill control.
4. Add a tactical layer.
5. Add opponents, then self-play.

## System Architecture

### Layer A: Motor Foundation

Responsibility:

- stable whole-body control at high frequency
- locomotion, recovery, contact timing, and ball interaction primitives

Inputs:

- proprioception
- optional short-horizon latent skill signal
- optional motion-tracking targets

Outputs:

- low-level action commands at policy frequency

Target properties:

- robust under disturbance
- contact-aware
- reusable across soccer and non-soccer tasks

### Layer B: Soccer Skill Adapter

Responsibility:

- turn generic body intelligence into football-specific action families

Inputs:

- motor-foundation latent state
- soccer task state
- skill parameters

Outputs:

- skill-conditioned latent or residual actions

Target properties:

- preserve motor prior
- adapt with small trainable heads first
- support kick, trap, dribble, pass, recover, intercept

### Layer C: Tactical Decision Model

Responsibility:

- select and parameterize the next behavior chunk

Inputs:

- ball state
- self state
- teammate states
- opponent states
- target geometry
- uncertainty and visibility

Outputs:

- skill type
- skill parameters
- hold / switch / abort signals
- short-horizon latent plan

Target properties:

- closed-loop
- object-centric
- robust under partial observability

### Layer D: Match-Scale Learning

Responsibility:

- convert isolated football skills into competitive behavior

Inputs:

- full multi-agent game state
- tactical objective

Outputs:

- long-horizon decision distribution over skills and positioning

Target properties:

- self-play compatible
- opponent-aware
- supports teamwork and pressure adaptation

## Interface Specification

The most important engineering step is to define interfaces before adding new models.

### Interface 1: `MotorState`

Minimal fields:

- base pose and velocity
- joint position and velocity
- projected gravity
- contact summaries
- previous action
- optional recurrent hidden state handle

Likely source:

- current policy observation terms in `tracking_env_cfg.py`

### Interface 2: `SoccerState`

Minimal fields:

- ball position in local frame
- ball velocity in local frame
- target / goal position in local frame
- strike-leg prior
- target destination
- confidence / noise proxy

Likely source:

- current `target_point_pos`
- current `target_destination_pos_local`
- future addition: explicit ball velocity observation

### Interface 3: `SkillCommand`

This should become the central contract between tactical layer and skill layer.

Minimal fields:

- `skill_type`: run, approach, trap, dribble, pass, shoot, recover
- `kick_leg`
- `approach_heading`
- `contact_time_hint`
- `contact_point_hint`
- `ball_velocity_target`
- `post_contact_heading`
- `recovery_style`

Short-term implementation:

- represent it as a structured tensor block appended to observations

Long-term implementation:

- explicit dataclass / manager term / command term

### Interface 4: `LatentSkillChunk`

Used only if diffusion / flow / chunked planner is added.

Minimal fields:

- latent token sequence or vector
- validity horizon
- optional contact schedule

Purpose:

- keep learned high-level generation separate from low-level action execution

## Current Repository Mapping

The following files are the most relevant execution surfaces in the current codebase.

### Training Entrypoints

- `scripts/rsl_rl/train_multi.py`
- `scripts/rsl_rl/play_multi.py`

These are the correct places to keep as outer orchestration entrypoints.

### Current Stage I / Stage II Environment Split

- `source/whole_body_tracking/soccer/tasks/tracking/config/g1/__init__.py`
- `source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py`

Current mapping:

- terrain motion tracking stage:
  `Tracking-Terrain-G1-RNN-v0`
- flat kick generalization stage:
  `Tracking-Flat-G1-SoccerDestination-RNN-v0`

### Observation and Reward Definitions

- `source/whole_body_tracking/soccer/tasks/tracking/tracking_env_cfg.py`
- `source/whole_body_tracking/soccer/tasks/tracking/mdp/observations.py`
- `source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards.py`
- `source/whole_body_tracking/soccer/tasks/tracking/mdp/terminations.py`

These files already encode most of the paper's Stage I and Stage II logic.

### Motion / Command Logic

- `source/whole_body_tracking/soccer/tasks/tracking/mdp/commands_multi_motion.py`
- `source/whole_body_tracking/soccer/tasks/tracking/mdp/commands_multi_motion_soccer.py`

These are high-priority files because they contain:

- multi-motion handling
- adaptive sampling
- soccer-ball placement
- target destination generation
- rolling-ball initialization

### Distillation Hook

- `source/whole_body_tracking/soccer/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `source/whole_body_tracking/soccer/tasks/tracking/config/g1/__init__.py`

A student-teacher config already exists. This is likely the cleanest nearby entrypoint
for experimenting with a foundation-model-to-soccer adaptation pipeline.

## Immediate Work Packages

### Work Package 0: Paper-To-Code Audit

Goal:

- produce a precise mapping between paper claims and repository implementation

Deliverables:

- which paper components are fully implemented
- which are approximated
- which are missing

Priority files:

- `soccer_flat_env_cfg.py`
- `commands_multi_motion_soccer.py`
- `rewards.py`
- deployment / exporter utilities

Success criteria:

- a written traceability matrix from paper section to file/function/config

### Work Package 1: General Motor Foundation Gap Analysis

Goal:

- determine how narrow the current Stage I is relative to a true sport foundation

Audit questions:

- what motions are currently included
- whether get-up / recovery / turn / sidestep exist
- whether there is explicit dribble / trap / pass data
- whether motion distribution is too kick-heavy

Deliverables:

- motion taxonomy
- missing-skill inventory
- proposed dataset expansion list

### Work Package 2: Skill Interface Refactor

Goal:

- introduce an explicit `SkillCommand` interface before changing model class

Short-term code direction:

- add structured skill-conditioning fields to the motion command
- expose them in observations
- allow fixed and scripted overrides for debugging

Deliverables:

- one explicit skill-conditioning contract
- one debug environment with manual skill control

Why now:

- this de-risks later tactical-model work

### Work Package 3: Rolling-Ball and Ball-State Upgrade

Goal:

- strengthen the current soccer state representation

Likely changes:

- explicit local-frame ball velocity observation
- ball confidence / visibility indicator
- short-horizon filtered state for deployment-consistent noise injection

Deliverables:

- upgraded `SoccerState`
- ablations on whether explicit velocity helps more than implicit recurrent memory

### Work Package 4: Tactical Layer Prototype

Goal:

- build the first decision model without yet requiring multi-agent play

Suggested version 1:

- object-centric MLP / transformer over structured state
- output `SkillCommand`
- initially train in 1v0 with delayed-shot / reposition scenarios

Not yet:

- no large VLA stack
- no raw-image end-to-end policy

Reason:

- first prove the interface and control decomposition

### Work Package 5: Generative Skill Planner Prototype

Goal:

- evaluate whether diffusion or flow-based middle-layer generation actually helps

Recommended order:

1. non-generative skill-parameter model baseline
2. chunked autoregressive baseline
3. flow-matching latent skill model
4. diffusion only if diversity clearly matters and latency is acceptable

Key metric:

- does generated planning outperform parameter regression under ball-motion variation and perturbation

### Work Package 6: Self-Play Soccer Curriculum

Goal:

- transition from isolated striker control to soccer behavior

Recommended sequence:

1. striker vs empty goal
2. striker vs goalie
3. striker vs defender
4. 2v2 pass-and-shoot
5. small-sided self-play

Critical infrastructure:

- opponent policy pool
- scripted opponents first
- metrics beyond raw reward

## Three-Month Plan

### Month 1

Focus:

- understanding and stabilizing the current repo

Tasks:

- complete the paper-to-code traceability document
- verify adaptive sampling implementation
- verify rolling-ball implementation
- verify sim-to-real components present vs missing
- inventory current motion dataset and skill coverage

Output:

- one audit report
- one missing-components report

### Month 2

Focus:

- preparing the repository for layered control

Tasks:

- introduce `SkillCommand`
- refactor soccer command pipeline to accept structured skill parameters
- add explicit ball velocity observation
- build scripted/manual skill-command tests

Output:

- first stable interface layer
- first debugging tools for high-level control injection

### Month 3

Focus:

- tactical layer v1 in 1v0

Tasks:

- implement a small tactical model over structured state
- compare against current motion-selection logic
- add task variants where direct shooting is suboptimal and repositioning is required

Output:

- first evidence whether high-level decision improvements help inside current repo constraints

## Six-Month Plan

### Months 4-5

Focus:

- foundation and generative planning exploration

Tasks:

- expand motion categories beyond kicks
- add recovery and turning data
- prototype adapter-based football specialization on top of a broader low-level policy
- benchmark regression vs flow-based skill generation

Output:

- broader motor prior
- first mid-layer planning benchmark

### Month 6

Focus:

- interactive soccer

Tasks:

- add goalkeeper or defender
- define multi-agent metrics
- start self-play curriculum with narrow scenarios

Output:

- first transition from isolated football skill to adversarial football behavior

## Metrics

Do not evaluate future work on shooting success alone.

### Low-Level / Skill Metrics

- motion tracking quality
- recovery success
- contact correctness
- kick direction error
- kick speed
- post-contact stability

### Decision Metrics

- shot timing quality
- success under delayed strike conditions
- ability to choose reposition over immediate bad shot
- time-to-shot under moving-ball scenarios
- re-touch / second-touch success

### Match Metrics

- possession retention
- expected shot quality proxy
- win rate in small-sided self-play
- robustness under opponent pressure

### Real-World Metrics

- deployment latency
- perception dropout robustness
- behavior under ball-parameter mismatch
- indoor/outdoor consistency

## Model Recommendations

### Tactical Layer

Use first:

- structured-state transformer or object-centric network

Do not use first:

- full image-language-action foundation model

Reason:

- current repo and problem stage do not justify full VLA complexity yet

### Mid-Layer Generator

Use first:

- flow matching if online latency matters

Use later:

- diffusion if multimodality and expressive motion chunk generation clearly justify slower sampling

### Low-Level Controller

Keep:

- recurrent, stable, high-frequency motor policy

Do not replace with:

- raw generative action model at control frequency

## Engineering Principles

- preserve small, reviewable diffs
- add interfaces before adding large models
- measure new layers against the current baseline before broad refactors
- keep sim-only ideas separate from deployment-ready code paths
- never blur tactical planning and low-level contact stabilization into one model unless a baseline proves it is worth it

## Suggested Next Tasks In This Repository

In recommended order:

1. Write the paper-to-code traceability document.
2. Audit `commands_multi_motion_soccer.py` against the paper's adaptive sampling and ball randomization claims.
3. Extract the current observation contract into a separate written spec.
4. Design and implement `SkillCommand v1`.
5. Add explicit ball velocity to observations and ablate against the current LSTM-only setup.
6. Build a tactical-layer sandbox task where the robot must choose between immediate kick and one-step reposition.

## Definition of Done For The Current Phase

The current exploratory phase is complete only when:

- the paper is fully mapped to code
- the current repo limitations are written down precisely
- the first explicit cross-layer interface exists
- one tactical prototype can control the existing low-level soccer policy through that interface

At that point, the project will have moved from "paper discussion" to "next-generation system construction".
