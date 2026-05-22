# Next-Gen Humanoid Soccer Roadmap

## Context

This document captures the working conclusions from the review of the paper
`Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework`
and the follow-up discussion about its limitations and the likely next research
direction for competition-level humanoid soccer.

The goal is persistence across future Codex/OMX sessions so the work can resume
from the same conceptual baseline instead of re-deriving the stack each time.

## Paper Summary

PAiD solves a narrow but valuable problem well:

- Stage I learns stable, human-like kicking primitives via motion tracking.
- Stage II adds lightweight perception and task rewards for ball-position
  generalization and rolling-ball handling.
- Sim-to-real focuses on contact-parameter alignment and structured observation
  noise.

The paper is strong as a `skill acquisition and transfer` work. It is weaker as
an actual `soccer intelligence` work.

## Main Critique

The core weakness is the high-level decision layer.

PAiD mostly learns a strong striker skill pipeline, not a competition-capable
soccer agent. The system is still close to:

- choose or condition on a motion
- approach the ball
- execute a kick

What it does not yet really solve:

- when to shoot vs delay vs reposition
- how to adapt continuously under pressure, disturbance, or opponent motion
- how to perform multi-touch behavior such as trap, dribble, recover, re-kick
- how to reason over teammates, defenders, space, and tactical value
- how to turn skill execution into match-winning behavior

In short:

- PAiD is a high-quality `shooting skill system`
- not yet a full `soccer decision-making system`

## Key Thesis For Next-Generation Work

The next system should not be "a better kicking policy".

It should be a `football agent stack` with explicit layering:

1. General motor foundation
2. Soccer skill adaptation
3. Tactical decision model
4. Multi-agent self-play and real-world adaptation

## Proposed Architecture

### Layer 1: General Motor Foundation

Stage I should be broadened from "kick tracking" to a general humanoid sport
foundation model.

Required coverage:

- locomotion: walk, run, accelerate, decelerate, cut, sidestep, turn
- stabilization: push recovery, stumble recovery, fall recovery, get-up
- contact organization: support-leg placement, swing timing, balance repair
- ball interaction primitives: trap, nudge, push, dribble, pass, shoot

This should be built from mixed data:

- motion tracking / mocap
- teleoperation or human demonstration
- disturbance curriculum
- recovery curriculum
- contact-rich soccer-specific motion data

Important principle:

- This is not only a locomotion foundation.
- It should be a whole-body sport foundation.

### Layer 2: Soccer Skill Adaptation

On top of the motor foundation, adapt to soccer.

Do not treat this as a library of fixed kicks. Instead learn a parameterized
skill space with variables such as:

- skill type
- kick leg
- approach angle
- contact timing
- contact point
- desired ball direction
- desired ball speed
- post-contact recovery heading

Preferred adaptation strategy:

- freeze most of the motor backbone
- use residual heads / adapters / LoRA-style specialization
- RL fine-tuning for football-specific execution

Avoid full end-to-end re-training from scratch because it risks damaging the
general body-control prior.

### Layer 3: Tactical Decision Model

This is the most important missing piece.

The tactical layer should operate on structured game state rather than only ball
and goal geometry. Inputs should include:

- ball state
- self state
- teammate states
- opponent states
- free-space / lane geometry
- uncertainty / visibility status

Outputs should not be raw joint actions. They should be:

- latent skill chunks
- skill parameters
- short-horizon intention variables
- switch / hold / abort decisions

This layer should answer:

- should I shoot now
- should I take one more control touch
- which foot should I use
- what outgoing ball velocity is best
- how should I position my body for the next state after contact

### Layer 4: Multi-Agent Self-Play

To become competition-capable, the system must move from 1v0 skill execution to
interactive soccer.

Recommended curriculum:

1. 1v0 skill execution
2. 1v0 with goalkeeper
3. 1v1 attacker vs defender
4. 2v2 passing and shooting
5. small-sided self-play

Important self-play features:

- historical opponent pool
- population-based training
- style diversity in opponents
- disturbance and partial observability

Without this, the system will remain skillful but not tactically robust.

## Where VLA, Diffusion, and Flow Matching Fit

### VLA

The project should use the spirit of VLA, not necessarily a literal
vision-language-to-joint-action pipeline.

Good use cases:

- high-level task conditioning
- semantic prompts over tactics or role
- multimodal fusion across ego vision, object state, and match context
- generalized high-level decision making

Bad use case:

- direct low-level whole-body control at humanoid control frequencies

Conclusion:

- VLA belongs in the tactical or high-level planning layer
- not in the final motor-control loop

### Diffusion

Diffusion is attractive for:

- multimodal skill generation
- motion prior modeling
- latent plan generation
- contact-rich short-horizon behavior synthesis

But direct diffusion-to-action at high rate is likely too slow and too fragile
for the final low-level controller.

Better use:

- generate latent motion chunks
- generate parameterized skill plans
- guide a lower-level stable controller

### Flow Matching

Flow matching is especially promising because it may preserve some of the
expressive benefits of diffusion while improving inference speed.

It is a strong candidate for:

- online skill-chunk generation
- short-horizon action-plan generation
- replacing diffusion when latency matters

Practical conclusion:

- use diffusion or flow at the middle layer
- prefer flow matching for online deployment-sensitive generation
- keep the final low-level controller deterministic/stable and high-frequency

## Most Important Research Principle

Do not build "a larger kick model".

Build a layered soccer agent:

- high-level: tactical reasoning
- mid-level: latent skill planning / chunk generation
- low-level: general humanoid motor foundation

This separation is likely the cleanest path toward both:

- more realistic soccer behavior
- higher match competitiveness

## Proposed Research Roadmap

### Phase A: Preserve and Extend The Current Repository

Use the current repository as the execution base for:

- paper-to-code mapping
- reward and observation analysis
- motion-skill pipeline inspection
- stage separation inspection

Immediate questions to answer in code:

- where the two training stages are separated
- how motion tracking is represented in observations and rewards
- whether adaptive sampling is fully implemented as described in the paper
- how rolling-ball support is implemented
- whether the physics-aware sim-to-real pieces exist fully in code

### Phase B: Build a General Motor Foundation

Expand Stage I into a reusable motor prior.

Deliverables:

- unified motor skill taxonomy
- broader motion dataset
- recovery and disturbance curriculum
- reusable low-level policy API

### Phase C: Replace Motion Retrieval With Parameterized Skill Planning

Replace hand-crafted or nearest-motion selection with a learned interface:

- latent skill representation
- parameterized kick / dribble / trap actions
- optional diffusion/flow planner for short-horizon skill chunk generation

### Phase D: Add Tactical Decision Making

Build a high-level decision layer that reasons over:

- geometry
- uncertainty
- opponents
- teammates
- shot value vs possession value

### Phase E: Move To Self-Play

Progress from striker skill to soccer behavior.

Metrics to prioritize:

- possession retention
- shot quality under pressure
- recovery after failed kick
- re-touch capability
- small-sided game win rate

## Practical Research Risks

Main technical risks:

- a foundation model that is too broad may dilute football-relevant contact skill
- a direct VLA-to-action approach may be too slow and unstable
- diffusion at the wrong layer may be impractical online
- full fine-tuning can destroy strong motor priors
- self-play without league/population structure may collapse to brittle behaviors

## Working Conclusions

Current best belief:

- the paper's strongest part is progressive decomposition
- the paper's weakest part is high-level decision making
- the next major leap requires replacing skill retrieval/execution dominance with
  layered decision + generation + robust motor foundation

The target system should be:

- not merely a better striker
- but a real soccer agent with reusable body intelligence

## Recommended Next Conversations

When resuming this work, continue with one of the following:

1. Map the paper's Stage I and Stage II claims to concrete repo code.
2. Audit whether adaptive sampling and sim-to-real components are implemented as
   written.
3. Draft a formal next-generation architecture diagram and module interface spec.
4. Turn the roadmap above into a staged engineering / research execution plan.
