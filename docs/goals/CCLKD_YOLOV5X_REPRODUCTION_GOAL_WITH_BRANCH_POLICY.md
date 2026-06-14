# CCLKD YOLOv5x Reproduction Goal

## 0. Objective

Complete an auditable YOLOv5x CCLKD reproduction on OGSOD-1.0 under:

```text
cclkd_reproduction/yolov5_sanity
```

The goal is to determine whether the current YOLOv5x CCLKD implementation correctly reproduces the paper-level mechanism and component trends.

This task is not about CoLD, CMDistill, YOLO11n, LADD, FGD, LD, HalluciDet, or other comparison methods.

Use server 90 for GPU experiments.

---

## 1. Current State Snapshot

Before doing anything, read the latest GitHub status under:

```text
cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/
```

The numbers below are historical context only. The authoritative status must be read from:

```text
cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/running_status.md
```

If current progress, stopping decisions, or new experiment decisions depend on the latest state, verify the runs directly on server 90 before acting.

Historical context from an earlier running snapshot:

```text
Full CCLKD:
  epoch = 185
  AP50 = 0.59976
  AP = 0.32287
  same-epoch det-only AP = 0.31722
  ΔAP = +0.00565
  KD/det ratio = 0.43622

ATKD-only:
  epoch = 151
  AP50 = 0.57164
  AP = 0.30139
  same-epoch det-only AP = 0.29451
  ΔAP = +0.00688
  KD/det ratio = 0.05804

CCL-only:
  epoch = 237
  AP50 = 0.62667
  AP = 0.35324
  same-epoch det-only AP = 0.34817
  ΔAP = +0.00507
  KD/det ratio = 0.40220

det_only_same_trainer:
  epoch = 268
  AP50 = 0.64535
  AP = 0.36632
```

Known facts:

```text
1. The custom trainer alignment issue has been fixed.
2. det_only_same_trainer is now aligned with standard YOLOv5 train.py.
3. two_branch_no_kd is close to det-only, so the two-branch setup itself is not the main problem.
4. The four 400epoch main runs are active.
5. Current paper runs show small positive same-epoch AP deltas.
6. CCL has high KD/det pressure but only small AP gain.
7. Full is positive vs det-only, but not yet clearly better than ATKD-only at common epochs.
```

Do not restart, duplicate, or replace these four main runs unless a run has crashed or produced invalid outputs.

---

## 2. Scope

Only work inside:

```text
cclkd_reproduction/yolov5_sanity
```

Allowed focus:

```text
YOLOv5x CCLKD reproduction
OGSOD-1.0
det-only baseline
paper_atkd_only
paper_ccl_only
paper_full
diagnostics
reports
compact archives
plots
same-epoch comparisons
```

Do not add:

```text
YOLO11n
CoLD reproduction
CMDistill reproduction
LADD experiments
FGD
LD
HalluciDet
new comparison methods
```

unless explicitly requested by the user.

---

## 3. Paper Alignment Targets

Use the CCLKD paper as the source of truth.

For OGSOD-1.0, the paper setting is:

```text
input size = 256 × 256
optimizer = SGD
epochs = 400
batch size = 32
learning rate = 0.01
momentum = 0.937
```

The key component target is the paper hyperparameter/component table:

```text
baseline:
  λkd = 0.0
  λcc = 0.0

ATKD-only:
  λkd = 1.0
  λcc = 0.0

CCL-only:
  λkd = 0.0
  λcc = 1.0

Full:
  λkd = 1.0
  λcc = 1.0
```

Paper-reported OGSOD target trend:

```text
baseline:    AP50 = 80.9, AP = 46.3
ATKD-only:   AP50 = 87.1, AP = 55.4
CCL-only:    AP50 = 85.8, AP = 54.5
Full:        AP50 = 88.7, AP = 57.3
```

Do not claim success only because AP delta is positive. Check whether the component trend is paper-consistent:

```text
ATKD-only > baseline
CCL-only > baseline
Full > ATKD-only
Full > CCL-only
Full is the best component setting
```

---

## 4. Required Main Experiments

The four required main runs are:

```text
det_only_same_trainer b32 400epoch
paper_atkd_only b32 400epoch
paper_ccl_only b32 400epoch
paper_full b32 400epoch
```

Always compare paper runs against det-only at the exact same epoch.

Do not compare:

```text
paper_full latest vs det-only latest
```

unless both are at the same epoch.

---

## 5. Current Main Job Policy

While the four main 400epoch jobs are active:

```text
Do not modify CCLKD loss formulas.
Do not modify PATM / ATKD.
Do not modify CCL.
Do not modify candidate source.
Do not add projection layers.
Do not change training protocol.
Do not launch broad sweeps.
Do not stop runs unless explicit stop criteria are met.
```

You may:

```text
monitor jobs
archive compact snapshots
generate plots
generate tables
generate milestone reports
grep compact logs
add analysis scripts
add offline diagnostic scripts
run small dry-runs
run small offline diagnostic jobs
```

Do not run broad new experiments before the 200/250 milestone decision.

---

## 6. Git Branch and Commit Policy

### 6.1 Never modify main directly for code changes

All code changes must happen on a dedicated branch. Do not commit code changes directly to `main`.

Use branch names like:

```text
codex/cclkd-yolov5x-diagnostics-YYYYMMDD
codex/cclkd-yolov5x-gradient-probes-YYYYMMDD
codex/cclkd-yolov5x-ccl-sweep-YYYYMMDD
codex/cclkd-yolov5x-reporting-YYYYMMDD
```

For example:

```bash
git checkout main
git pull --ff-only
git checkout -b codex/cclkd-yolov5x-diagnostics-20260613
```

### 6.2 Commit every logical code modification

Every code modification must be committed to GitHub. Use small, auditable commits.

Recommended commit format:

```text
scope: short description

Context:
- why this change is needed

Changes:
- files changed
- behavior changed

Validation:
- commands run
- dry-run or smoke result

Artifacts:
- included compact artifacts
- excluded large artifacts
```

Example:

```bash
git status
git add cclkd_reproduction/yolov5_sanity/tools/make_milestone_component_comparison.py
git commit -m "cclkd: add milestone comparison utility"
git push -u origin codex/cclkd-yolov5x-diagnostics-20260613
```

### 6.3 Evidence-only commits may use a separate evidence branch

Compact evidence updates may be committed either:

```text
A. to the current experiment/evidence branch; or
B. to a dedicated evidence branch
```

Recommended branch:

```text
codex/cclkd-yolov5x-evidence-YYYYMMDD
```

Evidence commits should not include code changes unless they are analysis/report scripts. If a commit mixes code and evidence, the commit message must explain why.

Evidence-only commits to `main` are allowed only when the user explicitly requests a direct GitHub update. Otherwise, use an evidence branch and wait for user approval before merging or opening a PR.

### 6.4 Running experiments must not be affected by branch changes

The four active 400epoch jobs must continue using the code commit they were launched from.

Do not:

```text
git pull
git checkout
git reset
pip install -e .
modify source files in-place
```

inside a working directory that is actively used by running jobs, unless the user explicitly confirms it is safe.

If code changes are needed while experiments are running, use one of these safe approaches:

```text
1. Work in a separate clone.
2. Work in a separate Git worktree.
3. Work on a separate branch in a directory not used by active jobs.
```

Recommended worktree example:

```bash
git fetch origin
git worktree add ../LADD_public_cclkd_diag codex/cclkd-yolov5x-diagnostics-20260613
cd ../LADD_public_cclkd_diag
```

### 6.5 Branch merge policy

Do not merge diagnostic or experimental branches into `main` unless the user explicitly asks.

Preferred workflow:

```text
branch -> commit -> push -> summarize -> wait for user approval
```

Optionally open a PR, but do not merge it automatically.

### 6.6 Large artifact policy

Never commit:

```text
.pt
.pth
TensorBoard event files
full nohup.log
large checkpoints
large images unless explicitly requested
large files over the agreed size limit
```

Use `.gitignore` or explicit `git status` checks to ensure these are excluded.

Before every commit, run:

```bash
git status --short
git diff --cached --check
git diff --cached --name-only | grep -E '\.(pt|pth)$|events\.out\.tfevents|(^|/)nohup\.log$' && exit 1 || true
find cclkd_reproduction/yolov5_sanity/results -type f \( -name "*.pt" -o -name "*.pth" -o -name "events.out.tfevents*" -o -name "nohup.log" \)
find cclkd_reproduction/yolov5_sanity/results -type f -size +5M
```

If large artifacts are found, do not commit them.

---

## 7. Milestone Monitoring

Archive snapshots at:

```text
150
200
250
300
350
399
```

For each snapshot, update:

```text
cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/running_status.md
cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/summary.csv
cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/milestone_component_comparison.csv
cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/milestone_component_comparison.md
```

Each milestone row must include:

```text
epoch
det_only_ap50
det_only_ap
atkd_ap50
atkd_ap
atkd_delta_ap50
atkd_delta_ap
ccl_ap50
ccl_ap
ccl_delta_ap50
ccl_delta_ap
full_ap50
full_ap
full_delta_ap50
full_delta_ap
full_minus_atkd_ap
full_minus_ccl_ap
best_component_by_ap
note
```

If a run has not reached the milestone epoch, write:

```text
pending
```

Do not use nearest epoch silently. Use exact epoch only.

---

## 8. Required Diagnostics

For every main run, track:

```text
AP curve
AP50 curve
same-epoch AP delta curve
same-epoch AP50 delta curve
student box/obj/cls losses
teacher box/obj/cls losses if applicable
kd_total_loss
LLD loss
FLD loss
RLD loss
CCL loss
weighted KD / student detection ratio
COP positive ratio
COP positive candidate count
COP valid candidate count
temperature mean/min/max
feature_capture_ok
nan_or_inf_detected
```

If available or feasible without disrupting the main runs, also collect:

```text
online teacher validation AP/AP50
teacher-student AP gap
```

If teacher checkpoint or validation artifact is missing, record it explicitly. Do not restart the main runs only to obtain teacher AP.

Gradient diagnostics are required for final explanation but must not contaminate the active main runs.

Gradient diagnostics should be implemented as separate offline scripts or small diagnostic jobs, such as:

```text
single batch or few-batch gradient norm probe
gradient cosine between detection loss and LLD
gradient cosine between detection loss and FLD
gradient cosine between detection loss and RLD
gradient cosine between detection loss and CCL
gradient norm for detection loss
gradient norm for ATKD
gradient norm for CCL
```

Do not add gradient hooks into the active 400epoch main jobs.

Before launching any new long ablation, prefer offline diagnostics from existing checkpoints:

```text
loss contribution analysis
candidate statistics
temperature statistics
teacher-student gap
gradient norm/cosine probes on one or a few batches
```

Use these diagnostics to decide whether a follow-up experiment is justified.

---

## 9. Decision Rules at 200/250/300/350/399

### Continue without changes

Continue current main runs if:

```text
Full >= det-only
Full >= ATKD-only
Full has meaningful positive AP gain
```

or if:

```text
Full > det-only
but component relation is not yet stable before 250
```

### Mark weak/negative CCL synergy

If:

```text
Full > det-only
but Full < ATKD-only
```

mark:

```text
CCL weak/negative synergy candidate
```

Continue to the next milestone unless stop criteria are triggered.

### Mark low-efficiency CCL

If:

```text
CCL-only has high weighted KD/det ratio
but weak AP gain
```

mark:

```text
CCL low-efficiency candidate
```

Do not immediately modify CCL. Wait for 250 or later unless the run becomes harmful.

### Stop or inspect Full

If:

```text
Full falls below det-only by more than 0.02 AP
```

then inspect diagnostics before continuing.

If:

```text
Full AP delta < -0.02
and weighted KD/det ratio is high
and diagnostics show no logging error
```

then stop Full and preserve artifacts.

### ATKD diagnostic trigger

At 250 or later, if any of the following holds:

```text
ATKD-only <= det-only
ATKD-only > det-only but delta AP < +0.02
ATKD-only is positive but far below the paper-level trend
ATKD weighted KD/det ratio remains very low while raw ATKD loss is nonzero
```

then mark:

```text
ATKD weak/low-transfer candidate
```

and prepare ATKD subcomponent diagnostics. Start with offline loss/gradient/candidate diagnostics from existing checkpoints before implementing new training modes.

Do not launch ATKD subcomponent experiments before this condition is met and the user approves the proposed experiment list.

---

## 10. Allowed Follow-up Ablations

Only after 200/250 milestone evidence justifies them.

### CCL sweep trigger

Only launch CCL sweep if:

```text
ATKD-only > det-only
Full <= ATKD-only
CCL-only weak or low-efficiency
Full weighted KD/det ratio much higher than ATKD-only
```

Then allowed experiments:

```text
paper_full with CCL weight 0.25
paper_full with CCL weight 0.5
paper_full with KD warmup 10
```

Do not launch these before the milestone condition is met.

All code or launcher changes required for sweeps must be made on a dedicated branch and committed before running.

### ATKD subcomponent trigger

Only if 250-or-later evidence indicates weak or non-paper-like ATKD behavior, such as:

```text
ATKD-only <= det-only
ATKD-only > det-only but delta AP < +0.02
ATKD-only is positive but far below the paper-level trend
LLD/FLD/RLD loss or gradient diagnostics suggest one subcomponent dominates or fails
```

Then propose, but do not automatically launch, these Table12-style incremental ATKD modes:

```text
paper_lld_only
paper_lld_fld
paper_lld_fld_rld_fixedT
```

Compare these against the existing `paper_atkd_only` run as the LLD+FLD+RLD+PATM setting. Do not duplicate `paper_atkd_only` unless the original run is invalid.

All new modes must be implemented on a dedicated branch, committed, pushed, smoke-tested, and approved before any experiment is launched.

---

## 11. Success / Failure Tiers

Use tiered conclusions.

### Strong reproduction success

Requires:

```text
Full > det-only with meaningful AP gain
Full > ATKD-only
Full > CCL-only
ATKD-only > det-only
CCL-only > det-only
training stable
diagnostics valid
trend is paper-consistent
```

Preferred meaningful AP gain:

```text
Full - det-only >= +0.02 AP
```

### Partial reproduction success

Use this if:

```text
Full > det-only
but gain is small
or Full is not clearly better than ATKD-only / CCL-only
or one component is weak
```

This means implementation has positive signal but does not reproduce the paper trend strongly.

### Explained reproduction failure

Use this if:

```text
Full <= det-only
or Full consistently < ATKD-only
or ATKD/CCL do not show expected component behavior
```

Failure explanation must include:

```text
trainer alignment evidence
det-only vs train.py evidence
two-branch no-KD evidence
teacher branch evidence if available
component diagnostics
loss scale diagnostics
KD/det ratio
COP statistics
temperature statistics
gradient norm/cosine diagnostics
paper comparison table
```

---

## 12. GitHub Evidence Policy

Commit only compact evidence:

```text
CSV summaries
Markdown reports
compact log tails
diagnostics CSV
generated figures
analysis scripts
process/GPU snapshots
```

Exclude:

```text
.pt
.pth
TensorBoard event files
full nohup.log
large artifacts
large images unless specifically needed
files over agreed size limit
```

Each evidence commit must include:

```text
scope
evidence included
validation performed
excluded artifacts
current decision
next action
```

---

## 13. Final Deliverables

The final deliverable is:

```text
cclkd_reproduction/yolov5_sanity/results/FINAL_CCLKD_YOLOV5X_REPRODUCTION_REPORT.md
```

It must include:

```text
1. experiment timeline
2. trainer alignment evidence
3. main 400epoch curves
4. same-epoch delta curves
5. component result table
6. original-paper vs ours comparison
7. diagnostics summary
8. teacher/student gap analysis if available
9. gradient diagnostic summary
10. success / partial success / failure conclusion
11. recommended next research decision
```

The final conclusion must explicitly answer:

```text
Can this CCLKD implementation be used as a valid comparison method in our paper?
Should we continue using the YOLO11n CCLKD adaptation?
Should we invest more GPU time in CCLKD, or stop and archive it as failed/partial?
```

---

## 14. Immediate Next Action

Do not change code.

Do not launch new experiments.

Continue the four active 400epoch jobs.

Generate exact-epoch milestone comparison tables at:

```text
200
250
300
350
399
```

The next important decision point is the 200/250 aligned snapshot.

If any analysis script must be added, create a dedicated branch, commit it, push it, and do not merge to main without user approval.
