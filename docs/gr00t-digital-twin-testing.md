# GR00T Digital Twin Testing: Current State and Gaps

*2026-11-08. Research summary behind the [test plan](gr00t-test-plan.md) and the digital-twin tooling in [twin failure repro](twin-failure-repro.md).*

## TL;DR

GR00T policy testing today is mostly closed-loop success-rate checks in Isaac Sim / Isaac Lab scenes, plus growing real-to-sim "digital cousin" research. It works well for checking capability and sweeping variations. It is still weak at showing that a sim result predicts real-robot performance, especially for humanoids, whole-body control and contact-rich tasks.

## What's done today

Current testing falls into seven layers, from per-checkpoint benchmarks to neural simulators.

1. **Standard sim benchmarks for each checkpoint.** N1.6 is evaluated on LIBERO (after 20–40k post-training steps), SimplerEnv (Google Robot fractal fine-tune), BEHAVIOR-1K, and IsaacLabEvalTasks for industrial manipulation such as nut pouring and exhaust-pipe sorting. Each released fine-tune is tied to one of these environments, from WidowX/Bridge to Unitree G1 loco-manipulation ([Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)).
2. **Isaac Lab-Arena, the main evaluation harness.** Co-developed with Lightwheel as an Isaac Lab extension, with APIs for task curation, variation generation and large parallel evaluation. Scenes compose Objects + Affordances + Scene + Embodiment + Task (e.g. a GR-1 opening a microwave). Built-in sweeps vary object, background and destination and report per-job success. Docs also cover sensitivity analysis, multi-node evaluation and agentic environment generation. NVIDIA's GEAR Lab uses Arena to benchmark the GR00T N models ([NVIDIA blog](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena), [Arena docs](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/first_experiments/running_a_real_policy.html)).
3. **Evaluation episodes generated from demonstrations.** The G1 loco-manipulation fine-tune expanded 10 teleoperated demos with MimicGen and was evaluated in Arena on dynamically generated pick-and-place episodes.
4. **Same server/client setup for sim and real.** Real-robot evaluation uses the same GR00T policy server; only the client changes (robot vs. simulator), which makes sim-vs-real A/B comparisons easy to wire up ([NVIDIA SO-101 course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/12-real-evaluation.html)).
5. **Real-to-sim "digital cousins" (mostly outside NVIDIA).** REALM compares a real DROID platform with its simulated cousin across about 800 rollout pairs and evaluates π0, π0-FAST and GR00T N1.5, concluding generalization and robustness remain open. PolaRiS and SIMPLER work the same way, scoring a sim by Pearson correlation and MMRV (how often sim and real disagree on policy ranking).
6. **Neural / world-model simulators.** RoboWorld rolls policies out inside a learned video world model and scores the generated trajectories. DreamZero, reportedly the basis for GR00T N2, already runs in Arena through a remote WebSocket inference client.
7. **Domain-specific twins.** Example: the healthcare workflow Rheo, covering twin composition, demo capture, synthetic data generation, policy training and pre-deployment validation.

## The gaps

The biggest gap is that few GR00T sim benchmarks are validated against real-world results; the other seven follow from it.

| # | Gap | Evidence |
| --- | --- | --- |
| 1 | Few sim benchmarks are validated against real results | REALM's authors say fully simulated evaluation of real-data policies is established only in SIMPLER. A June 2026 study found sim benchmarks are not widely adopted as reliable stand-ins for real evaluation. LIBERO is suspected of measuring memorization rather than generalization. |
| 2 | Correlation work covers tabletop arms, not humanoids | REALM, PolaRiS and SIMPLER are DROID/arm setups; SIMPLER-style methods cannot evaluate wrist-camera policies. Nothing comparable exists for G1 or GR-1 whole-body loco-manipulation. |
| 3 | Physics and actuator calibration is ad hoc | Practitioners report G1 passive joint stiffness near limits triggering torque limits, noisy real encoders vs. perfect sim joints, and contact-rich manipulation as the hardest case. No standard system-ID step ties a twin to the specific robot. |
| 4 | Metrics are thin | Mostly binary success over 10–20 episodes per condition (vs. 200 for reference scores). Few confidence intervals, failure taxonomies, recovery or safety metrics (contact force, self-collision, falls). Cross-report numbers often use different protocols. |
| 5 | Coverage is mostly visual and object variation | Arena sweeps objects, lighting and placement. Missing: a scenario language with coverage metrics, adversarial/falsification search, long-tail mining, and a path to a safety case. |
| 6 | Timing is not tested as a system | Sim steps the policy in lockstep; real deployment runs async action chunks on Jetson Thor (DreamZero: 150 ms per chunk, 7 Hz control). Latency, jitter and dropped frames are rarely injected. |
| 7 | Neural simulators bring their own risks | World-model evaluators can hallucinate physics, so scores may reflect evaluator artifacts. A world-action model graded by a video world model from the same family may share its blind spots. |
| 8 | Other domains are thin | Deformables, tactile sensing, humans in the scene and multi-robot setups are barely covered. |

## Comparison with AV validation

Robotics policy evaluation is roughly where AV was before scenario-based testing. Mature AV simulation validation has four things largely absent here:

- A scenario description layer with coverage criteria tied to the robot's intended operating domain
- Simulation credibility assessed per scenario class, not one global sim-real correlation number
- Replay of real field logs back into the twin
- A statistical argument behind release decisions

Each is a gap a team could own on top of Arena, which already has hooks for variations and sensitivity analysis.

## Sources

- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [NVIDIA blog: Isaac Lab-Arena](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena)
- [Isaac Lab-Arena docs](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/first_experiments/running_a_real_policy.html)
- [REALM (arXiv 2512.19562)](https://arxiv.org/html/2512.19562)
- [PolaRiS (arXiv 2512.16881)](https://arxiv.org/pdf/2512.16881)
- [Sim-and-Real Correlation for VLA Evaluation (arXiv 2606.10366)](https://arxiv.org/abs/2606.10366)
- [ETH RC: GR00T N1.6 zero-shot vs fine-tuned](https://ethroboticsclub.github.io/robosec-embodied-ai/experiment_1/1.1/)
- [NVIDIA SO-101 sim-to-real course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/12-real-evaluation.html)
- [DreamZero project page](https://dreamzero0.github.io/)
