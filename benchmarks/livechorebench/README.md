# LiveChoreBench: A Live Real-World Benchmark for Humanoid Robots

*2026-11-10. Design proposal with a runnable starter kit: task specs, a loader and scorer, sim and real-room adapters, an audited AI judge, required three-level logging, and tests.*

## In this folder

| Path | What it is |
|---|---|
| `tasks/*.yaml` | The eight sample tasks from section 2b, one spec each, runnable in sim and in a real room |
| `livechorebench/spec.py` | Loads and checks specs, samples instances from a seed (sim randomizer and real placement sheet), and scores a finished episode from a world state |
| `livechorebench/world.py` | Shared rules that turn poses and readings into the world state: region boxes, targets, stillness, drops, falls, lift height, route progress |
| `livechorebench/adapters.py` | `IsaacLabAdapter` (reads `env.unwrapped.scene`) and `RealRoomAdapter` (AprilTags or mocap, encoders, scales, robot state, speech log) |
| `livechorebench/judge.py` | AI judge for yes/no questions sensors can't answer, with any VLM backend, and its audit against human grades |
| `livechorebench/logs.py` | Required policy, action and joint logs, and the checker that rejects episodes with incomplete logs |
| `cells/example_cell.yaml` | A real-room cell file: surveyed fixture poses, tag IDs and offsets, scales |
| `tests/` | Specs load and validate; sampling is seeded; scoring, partial credit and grading are correct; sim and real adapters give the same world and score; the judge parses, abstains and is gated by its human audit; incomplete logs invalidate an episode |

Run the tests:

```bash
pytest benchmarks/livechorebench
```

They are kept apart from the main `tests/` suite and depend only on PyYAML and NumPy. The judge's Anthropic backend needs the `anthropic` package; tests use a fake backend.

Use it from a simulator or a real-room runner:

```python
from livechorebench import load, sample, score, placement_sheet

spec = load("benchmarks/livechorebench/tasks/e1_mug_to_rack.yaml")
inst = sample(spec, seed=42)        # same instance in Isaac Lab and on the real placement sheet
# ... run the episode, then build a world state from sim state or from tags/mocap/scales/encoders ...
result = score(spec, world_state)   # success, partial credit, per-subgoal results
print(placement_sheet(spec, [42, 43, 44]))
```

The world-state format is documented at the top of `spec.py`. Both adapters produce it, and the same scoring code runs on both.

### Adapters: sim state or real sensors to the scorer

```python
from livechorebench.adapters import IsaacLabAdapter, RealRoomAdapter

sim = IsaacLabAdapter(spec, env_index=0)            # one per parallel env
world = sim.read(env, t=env_time)                    # call every control tick

real = RealRoomAdapter(spec, "benchmarks/livechorebench/cells/example_cell.yaml")
world = real.update({"t": 12.3, "tags": tag_poses, "encoders": {...}, "scales": {...},
                     "robot": {"base_height_m": 0.81}, "speech": [...]})
```

The geometry that decides success lives in the spec, not in either adapter: `scene.regions` are boxes in an object's frame (`rack.slots`, `drawer.volume`), `scene.targets` are poses in an object's frame (place settings on the mat), and `scene.route` is the path for progress. Spec loading refuses any region, target or joint a predicate uses but doesn't declare, so a task can't pass in sim on geometry the real room can't measure. The shared `WorldBuilder` computes:

| World field | Sim source | Real source |
|---|---|---|
| Object poses, tilt, yaw | `scene[name].data.root_pos_w`, `root_quat_w` | AprilTag (object centre from the tag offset) or mocap; last pose kept while occluded |
| Fixture poses | Same as objects | Surveyed once into the cell file |
| `inside` a region | Pose inside the declared box | Same, or a scale under the region whose net weight proves which objects are on it (a closed drawer hides tags) |
| Joints (drawer slide) | `joint_pos` by the object's `joint` name | Encoder named by the object's `real_joint` |
| `<object>.height_m`, `route.progress` | From poses | From poses |
| `stable_s`, `dropped`, `fallen` | Speed of movable objects, fall below 0.2 m from above it, robot base height | Same rules on measured poses and the robot's state estimate |
| `said` | Robot's speech/text output | Speech log |

### AI judge

Some outcomes aren't instrumented: whether a table *looks* set, or a towel is folded neatly. A spec asks these as yes/no questions with the `judge` predicate (the set-table task has one). Rules:

- **Measured beats judged.** `judge` may appear in subgoals and grades but never in `success`; spec loading refuses it there.
- **Abstain rather than guess.** The judge answers `true`, `false` or `null` from end-of-episode frames; `null`, or confidence under the threshold, counts as not passed.
- **Any model.** `Judge(backend)` takes any `backend(prompt, images) -> text`; `anthropic_backend()` is provided.
- **Audited before trusted.** `agreement(judge, human)` reports agreement with a Wilson interval, Cohen's kappa, coverage, false passes and false fails on human-graded episodes. `audit_gate` lets a judge grade unsupervised only if the agreement lower bound is ≥ 0.85, kappa ≥ 0.7, it answers ≥ 90% of items, and false passes are ≤ 5%.

### Required logging: policy, action and joint level

Every spec declares a `logging` block, and an episode whose logs fail `check_episode_log` does not count. The three levels let any result be traced from the model's input to what every joint did:

| Level | Default rate | Required fields | Why |
|---|---|---|---|
| Policy | 10 Hz (every query) | `t, step, model_id, checkpoint_hash, instruction, obs_hash, action_chunk, latency_ms` | Which model and input produced which chunk; re-query and replay |
| Action | 50 Hz (100 Hz loco) | `t, policy_step, commanded, applied, clipped` | Commanded vs applied after limits and smoothing; links to the policy step |
| Joint | 200 Hz (500 Hz loco, plus `base_pose`, `foot_contact`) | `t, position, velocity, effort` | What the body actually did; triage and sim-vs-real comparison |

Spec loading requires all three, with minimum rates (1, 10, 50 Hz) and policy ≤ action ≤ joint. The episode checker rejects missing streams or fields, unequal column lengths, non-increasing time, rates below 90% of the declared rate, actions that name an unlogged policy step, and joint logs that don't cover the action span. Gaps over three periods and more than 10% clipped commands are warnings. These logs are what `groot-eval triage` and the twin track consume.

## Summary

This is a humanoid version of LiveCodeBench. Three of its ideas carry over:

- **Fresh tasks:** chores are created after a model's training cutoff, so they can't be memorized.
- **Automatic verification:** every chore is checked against a written rubric.
- **Multiple attempts:** models get several tries, scored like pass@k.

For a humanoid, "fresh problems" becomes "chores the robot has never seen, issued on the fly in real homes or labs".

## 1. Where the chores come from (staying "live")

- **Composed at test time, not from a fixed list.** A generator samples a chore template (tidy, fetch, load, wipe, fold, sort, set the table), then objects, locations and constraints. An LLM writes the instruction in natural language, sometimes ambiguous on purpose.
- **Date-stamped releases.** Each month adds new chore types, new objects and new phrasings. A model is scored only on releases after its training cutoff. This is LiveCodeBench's contamination control.
- **Rotating real objects.** A purchase list refreshed every quarter, so the objects aren't in any training set. Use real homes or apartment-style labs, not one fixed kitchen.
- **Hidden test set.** Only past releases are public, so labs can't tune on what they'll be graded on.

## 2. Tracks

These mirror LiveCodeBench's scenarios.

| Track | What it tests |
|---|---|
| Execute | Do the chore from a single instruction |
| Self-repair | After a failure, the robot gets feedback ("the cup is still on the table") and retries |
| Clarify | Ambiguous instructions: does it ask, or pick a sensible reading? |
| Long-horizon | Multi-step chores of 5–20 minutes, e.g. "clean up after dinner" |
| Disturbance | Someone moves an object or bumps the robot mid-task |
| Locomotion + manipulation | Chores that need walking between rooms and stairs while carrying |

## 2a. One task spec for sim and real

Every chore is written once, as a spec that both an Isaac Lab environment and a real-room runbook read. A task translates to sim and to real when everything that decides success is stated as **object poses, joint states or sensor readings**, never as a judgement call.

Each spec has six parts:

- **Scene:** each object as a USD asset for sim and a purchasable item for real (same dimensions and mass), plus fixtures (counter, rack, drawer) at surveyed positions.
- **Variation:** the same numeric ranges in both. In sim they are sampled by the randomizer from the seed. In the real room the same seed prints a placement sheet: a grid template or jig, with each object's position and yaw.
- **Instruction:** the text, generated per instance (paraphrases are part of the variation).
- **Success predicate:** a function of object and joint states, e.g. "mug inside rack volume, tilt < 20°, held 2 s". In sim it reads simulator state. In the real room it reads the same quantities from AprilTags or a motion-capture system, an overhead camera, scales and drawer encoders.
- **Events:** disturbances triggered by task phase (e.g. "when the gripper is within 20 cm of the mug, move the mug 15 cm"). In sim this is a scripted event. In the real room a tracker gives the cue and a person or a linear stage acts on it.
- **Limits:** timeout, safety limits, and what counts as an intervention.

See `tasks/e1_mug_to_rack.yaml` for a full example. The same predicate code scores sim and real. A sim/real disagreement is then a disagreement about the world, not about the grader.

## 2b. Sample tasks by track

Each task lists its sim setup (Isaac Lab or similar), its real setup, how success is checked, and the metrics specific to it. The general metrics in section 3 apply to every task. The spec file is named after each task.

### Execute

**E1. Put the mug in the dish rack.** (`tasks/e1_mug_to_rack.yaml`)
- **Sim:** tabletop scene with mug and rack assets; mug position ±10 cm, any yaw, three mug models, lighting 200–800 lux.
- **Real:** the same rack fixed at a surveyed spot; the mug placed from the printed placement sheet; an AprilTag on the rack, an overhead camera, and a load cell under the rack.
- **Success:** mug inside a rack slot, tilt under 20°, still for 2 s.
- **Task metrics:** placement error (cm), grasp attempts, drops.

**E2. Put the three items on the counter into the drawer and close it.** (`tasks/e2_items_to_drawer.yaml`)
- **Sim:** articulated drawer (prismatic joint), three objects drawn from a pool of 10, placed anywhere on the counter.
- **Real:** the same drawer with a string encoder or tag on its front; objects tagged or found by the overhead camera.
- **Success:** all three items inside the drawer volume and the drawer closed to within 1 cm.
- **Partial credit:** 1 point per item in the drawer, plus 1 for closing it (out of 4).

### Self-repair

**S1. Put the mug in the dish rack (with a forced first failure).** (`tasks/s1_mug_forced_failure.yaml`)
- **Sim:** friction on the mug's handle drops for the first grasp only, so the mug slips back onto the table.
- **Real:** a mug with a slippery sleeve that a person removes after the first grasp, or a person nudges the mug out of the gripper.
- **Feedback:** after the failure, the robot is told "The mug is still on the table."
- **Task metrics:** recovered (yes/no), attempts to recover, time from failure to success.

### Clarify

**C1. "Bring me the cup" with two cups on the counter (red and blue).** (`tasks/c1_which_cup.yaml`)
- **Sim and real:** two cups of different colours, both within reach; the person's position is fixed and marked.
- **Scoring:** full marks for asking which cup, or for choosing by a stated rule (e.g. the nearer one) and saying so. Half marks for delivering either cup silently; zero for anything else.
- **How measured:** the robot's speech or text log, plus the final cup position.

### Long-horizon

**L1. Set the table for two.** (`tasks/l1_set_table.yaml`)
- **Sim:** two place settings (plate, fork, cup each) from a cupboard and drawer; target spots marked on the table.
- **Real:** the same table with a printed mat showing the six target spots; items tagged.
- **Success:** all six items within 3 cm and 15° of their spots.
- **Partial credit:** 1 point per correctly placed item (out of 6).
- **Task metrics:** time per item, order of actions, items moved more than once.

### Disturbance

**D1. Mug to rack, with the mug moved mid-task.** (`tasks/d1_mug_moved.yaml`)
- **Sim:** when the gripper comes within 20 cm of the mug, the mug is moved 15 cm in a random direction.
- **Real:** a tracker raises the same cue and a person (or a linear stage) moves the mug by the same amount, using a marked path.
- **Task metrics:** success after the move, time to re-grasp.

**D2. Carry a tray across the room, with a push.** (`tasks/d2_tray_push.yaml`)
- **Sim:** a 40–60 N lateral push to the torso for 0.2 s, mid-walk.
- **Real:** a padded push sled or pendulum giving the same impulse, triggered at the same point on the path.
- **Task metrics:** fall (yes/no), items dropped, recovery time, peak tray tilt.

### Locomotion + manipulation

**M1. Carry the laundry basket from the bedroom to the washer.** (`tasks/m1_laundry_basket.yaml`)
- **Sim:** a 6–8 m route through one doorway and over a 2 cm threshold; basket mass 2–5 kg.
- **Real:** the same route marked on the floor of a real apartment or mock-up; basket weighed before each run.
- **Success:** basket on the marked area by the washer, nothing dropped, no fall.
- **Task metrics:** time, path length vs shortest path, contacts with door frames, energy used.

## 2c. Measurement plan

### How each metric is measured

| Metric | In sim | In the real room |
|---|---|---|
| Success, partial credit | Success predicate on simulator state | Same predicate on tag/mocap poses, scales, encoders; VLM judge on camera video for anything not instrumented |
| pass@1, pass@k | Episode outcomes per attempt | Same, with a standard reset between attempts |
| Time to completion | Sim time from instruction to predicate first true | Wall time from the logged instruction to the predicate first true |
| Placement accuracy | Pose error from simulator state | Pose error from tags or mocap |
| Human interventions | Not applicable (counts scripted resets) | Operator presses a logged button for each assist, reset or takeover, and picks a reason code |
| Collisions, contact force | Contact sensors and forces in physics | Robot's joint torque and force sensors, plus a bumper or force mat where fitted |
| Falls | Base height or orientation past a threshold | Same from the robot's IMU and state estimate; confirmed on video |
| Drops and damage | Object fell below table height | Object tracking plus a post-run check; damaged items logged |
| Instruction fidelity, clarification | Dialogue log vs rubric | Same, plus audio transcript |
| Efficiency | Energy from joint power, path length | Battery draw from the robot, path from the state estimate |

Every episode records the seed, task spec version, robot profile version and full traces, so any result can be replayed in the twin.

### How often

| Activity | Frequency | Size |
|---|---|---|
| New task release | Monthly | 10–20 new task specs and new instruction paraphrases |
| Object rotation | Quarterly | Replace about a third of the object pool |
| Model evaluation, sim | Per model per release | ~1,000 instances per task (cheap; gives tight intervals) |
| Model evaluation, real | Per model per release | ~100 episodes: 30 Execute, 15 each for the other tracks, sized for about ±10 points at 95% |
| Control run (reference policy) | Daily, per site | 2 tasks × 10 episodes; a drop outside its usual band pauses the site |
| Calibration-day checks | Weekly per robot, and after any repair | Contract checks, camera pose, robot profile replay error |
| Judge audit | Weekly | Humans grade a random 10–20% of real episodes; publish judge agreement |
| Sim-vs-real credibility | Per release | Per track: rank correlation, MMRV and paired outcome agreement between sim and real |

### What gets reported

Per model and per track:

- pass@1 and pass@3 with intervals;
- mean partial credit;
- median time to completion;
- interventions per hour;
- safety events per 100 episodes.

Also, per track, the sim-vs-real verdict (trusted, provisional or not trusted). It tells readers which sim numbers they can rely on.

## 3. Metrics

Each chore gets a written grading rubric in advance: its subgoals, success conditions and forbidden outcomes.

- **pass@1 / pass@k:** success on the first attempt, and within k attempts with a reset between them.
- **Attempts to success:** how many attempts it took, plus the robot's own retries and recoveries within an episode.
- **Partial credit:** share of subgoals completed (e.g. 4 of 5 dishes put away). This separates "almost" from "nothing".
- **Accuracy and quality:** right object, right place, orientation within tolerance, nothing left out. Scored from the rubric.
- **Time to completion:** reported as a distribution. Timeouts count as "not finished" rather than being dropped.
- **Human interventions:** assists, resets and teleop takeovers per hour. This is the real-world cost, and it matters most.
- **Safety:** collisions, contact force over a limit, falls, objects dropped or damaged, near-misses with people. Any severe event is a hard fail.
- **Instruction fidelity:** did it do what was asked, and nothing it wasn't asked to do?
- **Efficiency:** energy used, distance walked, wasted motion.

Report everything with confidence intervals. Where possible, rank with pairwise comparisons: two models on the same chore in the same room. This also lets you build Elo-style rankings across many sites.

## 4. Grading

- **Automated first:** before/after scene checks from fixed cameras plus object tags or scale sensors, with a VLM judge scoring against the rubric.
- **Validate the judge:** humans grade a random 10–20% of episodes. Publish the judge's agreement with humans, and don't trust a judge that hasn't been measured.
- **Blind graders:** reviewers don't know which model ran.

## 5. Making real-world results comparable

This is the hard part, and where most robotics benchmarks fall down.

- **Standard kit per site:** reference objects, lighting checks and a camera layout. Every robot passes calibration-day checks: contract checks and robot profiles.
- **A daily control run:** a fixed reference policy runs the same chores every day. If its score drifts, the site or the robot drifted, not the models.
- **Paired design:** each model runs the same chore instances under the same conditions, so you can use paired statistics.
- **Hardware normalization:** report results per robot model. Optionally add an embodiment-agnostic score from a shared robot pool.
- **Logging everything:** seeds, layout photos, the robot profile version and full traces, so any failure can be replayed and triaged later.

## 6. A twin track

Run every chore release in a digital twin of each site as well. Then the benchmark can publish:

- **Sim-vs-real agreement per chore class:** which chore types sim can predict and which it can't. This uses grooteval's twin credibility verdicts (`groot-twin credibility`).
- **Failure attribution for real failures:** plumbing, timing, actuation, sensing, policy, or something the twin doesn't model. This uses grooteval's failure triage (`groot-twin triage`).

No current benchmark does this. It would make LiveChoreBench the reference for "does sim predict real?".

## 7. What makes it hard

- **Cost and throughput:** real robot time is slow. Budget roughly 50–100 episodes per model per release, sized from the interval width you need.
- **Resets:** these need staff or auto-reset stations, and the reset procedure itself must be standardized.
- **Safety and liability** in homes, especially with people nearby.
- **Fairness across hardware:** a stronger robot body can hide a weaker model.
- **Grader drift:** keep validating the automated judge against humans.

## One-line pitch

> A LiveCodeBench for humanoids: date-stamped chores generated after the model's training cutoff, run in real rooms with standard kits and a daily control policy. It scores pass@k, partial credit, time to completion, interventions per hour and safety events, all with intervals. Each chore is paired with its digital twin, so the benchmark also measures where simulation can stand in for real tests.

## What is not here yet

- **Live hardware glue:** the adapters take poses and readings; the AprilTag detector, mocap client, scale and encoder drivers, and the ROS 2 node that feeds them are not included. The Isaac Lab adapter is tested against a fake scene, not a running simulator.
- **Log writer:** the checker defines the format; a recorder (MCAP or LeRobot) that writes it is not included.
- **Task generator:** the template and LLM generator for monthly releases.
- **Judge calibration data:** the judge and its audit are here; the human-graded episodes to audit it with are not.
- **Twin credibility per track:** connecting the twin track to `groot-twin credibility`.
- **Asset and SKU names:** the ones in the sample specs are placeholders.
