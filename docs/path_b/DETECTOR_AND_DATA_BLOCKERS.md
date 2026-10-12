# Detector and Training-Data Blockers

Status: **must be resolved before `PIPELINE_EVOLUTION.md` proceeds past step 3** (the
clip-classifier milestone, §5 step 4, and the `POSSESSION`/`TURNOVER` live rollout, §5
step 2). This doc expands on `PIPELINE_EVOLUTION.md` §7 items 1, 4, and 5 — those items
now point here for detail. Written 2026-09-17.

**Decision (2026-09-23): `yolov8n_gsr.pt` (Path A, the 4-class detector with
`player`/`goalkeeper`/`referee`/`ball` classes baked in) is stale.** Work going forward
is on **Path B only**: the athlete/ball detector (`runs/yolo26s_athlete_ball`,
2 classes — `athlete`, `ball`) plus `gsr_pipeline/athlete_roles.py`'s clustering to
recover `player_team_a/b`, `goalkeeper_team_a/b`, `referee`, `other` downstream. §1–§2
below (the `yolov8n_gsr` writeup and the referee/ball error analysis on it) describe
Path A and are kept for historical reference only — do not use them to justify new
work on `yolov8n_gsr`. §5 (Path B's role/team classifier) is the live section; new
detector/role work should extend it, not §1–§2.

**This is a directional decision, not yet an implemented switch (confirmed 2026-09-24):**
`gsr_pipeline/configs/default.yaml`'s `detector.model_path` still points at `yolov8n_gsr.pt`,
and `detect.py` still wraps that 4-class model as the one actually loaded by
`GSRPipeline`/`process_stream`. `athlete_roles.py` (Path B's role/team classifier) is
implemented and scored offline (§5) but is **not wired into the pipeline** — see
`AGENTS.md`'s note on `gsr_pipeline/athlete_roles.py`. Swapping the live detector to
`runs/yolo26s_athlete_ball` and wiring `athlete_roles.py` in is itself the
next piece of unstarted work this decision points at.

---

## 1. The detector, in simple terms

**What it is.** One model does all four object types at once — there is no separate
ball detector. It's YOLOv8n (the smallest/fastest version of the YOLOv8 object detector,
~6MB), fine-tuned to recognise 4 classes instead of YOLO's normal 80 COCO classes:

| class_id | role |
|---|---|
| 0 | player |
| 1 | goalkeeper |
| 2 | referee |
| 3 | ball |

- **Code:** `cv_service/gsr_pipeline/detect.py` (`YOLOv8nDetector`) wraps it;
  `cv_service/gsr_pipeline/train_yolo.py` produced it.
- **Weights file:** `yolov8n_gsr.pt`, committed at the `cv_service` repo root (~17.5MB).
  Loaded via `configs/default.yaml`'s `detector.model_path`.
- **Dataset fine-tuned on:** SoccerNet-GSR (`gamestate-2024`), converted from its
  `Labels-GameState.json` per-sequence annotation files into YOLO's `.txt` label format
  by `train_yolo.py`'s conversion step. Trained on the `train` split, model-selected
  ("best checkpoint") on an unseeded random 65% slice of the `valid` split
  (`GSR_Subset/GSR_Subset_65` per `train_yolo.py`'s `ROOT_DIR`) — meaning the 65% number
  is just which chunk of the valid split happened to get used, not a deliberate holdout
  design.
- **Fine-tune settings** (`train_yolo.py`, `TRAIN_CFG`): 50 epochs, image size 640px,
  batch size 64, early stopping after 10 epochs without improvement, starting from
  Ultralytics' stock `yolov8n.pt` pretrained backbone (COCO weights, replaced classification
  head for the 4 GSR classes).
- **Confidence thresholds at inference** (`configs/default.yaml`): 0.25 for
  player/goalkeeper/referee, a lower 0.15 for ball (already loosened once because the
  ball is the hardest class).

**Measured quality** (`cv_service/gsr_pipeline/evaluate_detection.py`, one 750-frame
sequence, `SNGS-021` — see `PIPELINE_EVOLUTION.md` §6 for the full table): player/
goalkeeper detection is good (95–99% recall). Referee detection has a real 18%
error rate. **Ball detection is the severe one: 52.4% recall — the ball is missed in
about half of all ball-labelled frames**, with gaps up to 168 consecutive frames (6.7s).

**Why the ball specifically is bad.** The ball is tiny — about 10–25 pixels across in a
1920×1080 broadcast frame. YOLOv8n was trained and runs inference at `imgsz=640`, i.e.
the frame is shrunk ~3x before the model ever sees it, so the ball shrinks to roughly
3–8 pixels — close to invisible at that scale. This isn't a training-data-volume
problem, it's a resolution problem: the model is looking at the frame too small to see
the ball reliably no matter how much data it's trained on.

## 2. How the detector can be improved

Options, roughly cheapest-to-try first:

1. **Raise inference resolution for the ball specifically.** Re-run detection at a
   larger `imgsz` (e.g. 1280 instead of 640) so the ball occupies more pixels. Costs
   more GPU time per frame; can be applied to the whole frame or only worth it if
   ball recall improves enough to matter.
2. **Tiled/sliced inference around the last known ball position.** Once the ball has
   been seen once, crop a smaller high-resolution window around its last position each
   subsequent frame and run detection on that crop at full resolution, falling back to
   full-frame search when it's lost. Cheaper per-frame than raising resolution
   everywhere, but adds tracking-recovery logic.
3. **A dedicated ball detector/tracker**, separate from the 4-class model — a model
   whose only job is finding a small round object, potentially combined with a motion-
   based tracker (e.g. predicting trajectory through occlusion/motion blur) rather than
   re-detecting from scratch every frame.
4. **Retrain at a larger `imgsz`.** The model was trained at 640; retraining at 1280
   (more expensive training, more expensive inference) may itself improve ball recall
   independent of any inference-time trick, since the model would learn ball features
   at a scale closer to what real ball pixels look like.

### Results so far (2026-09-21)

Options 1 and 4 have now been tried. `SNGS-021`, all 750 frames, CPU, scored like
`evaluate_detection.py` (people class-agnostic at IoU 0.5, ball within 15px, one-to-one
matching). The kickoff (frames 0–167: ball on the centre spot, white on white, half hidden
behind a player's legs) is missed by **every** model at every threshold, so the ball columns
are **open play only** (frames 168–749). People columns are whole-sequence at conf 0.25.

| model @ inference size | CPU s/frame | people recall / precision | ball recall / precision (conf 0.15) | longest ball gap |
|---|---|---|---|---|
| `yolov8n_gsr` @640 (current) | 0.11 | 94.7% / 94.5% | 67% / 79% | 31 frames |
| `yolov8n_gsr` @1280, not retrained | 0.31 | 93.6% / 85.0% | 49% / 77% | 51 |
| yolo26n @640, 4-class (`gsr_yolo26` stage 1) | 0.09 | 95.3% / 95.3% | 56% / 71% | 18 |
| yolo26n-p2 @640, 4-class (`gsr_yolo26` stage 2) | 0.12 | 95.4% / 95.2% | 67% / 63% | 21 |
| yolo26s athlete/ball @640 | 0.22 | 96.0% / 95.9% | 65% / 75% | 20 |
| yolo26s athlete/ball @1280 (`runs/yolo26s_athlete_ball`) | 0.73 | 97.0% / 95.5% | 79% / 71% (77% / 78% at conf 0.25) | 22 |

- **Option 1 is a dead end.** Raising inference resolution on the existing model makes it
  worse (ball recall 67% to 49%, person precision 94% to 85%): it needs to be trained at the
  higher size.
- **Option 4 works.** Training at 1280 gives about +10 points of open-play ball recall at
  matched precision (77% vs 67%, both near 78–79% precision) and a shorter longest gap.
- **The gain is resolution, not capacity.** The same yolo26s at 640 only matches the
  current ball recall, and the nano P2 head at 640 did not beat it either.
- **Cost.** About 6.6x CPU time per frame and roughly 10x FLOPs versus the current model. On
  a big GPU small models are launch-bound, so the gap should shrink to about 2–3x
  (an estimate from Ultralytics' T4 benchmarks, not measured here).
- **It has no roles.** The 2-class model cannot tell players, goalkeepers and referees
  apart. See §5.
- **Caveats.** One 30-second clip, from the `valid` split, so checkpoint selection may have
  seen it (true of the baseline too). Re-run on more sequences before deciding.

Options 2 and 3 (tiled inference, a dedicated ball tracker) are still untried. The
kickoff case (a static, occluded, white-on-white ball) is a different problem from detection
quality and is not fixed by any option here.

---

## 3. Geometric baseline (§7 item 4) — do we have ground truth to measure it against?

**Short answer: no usable ground truth exists yet for this — building it is the same
work as §4 below (training labels), not a separate problem.**

To measure whether a cheap rule-based guesser (using only ball speed/direction/
possession, no video) can already tell PASS from SHOT from CARRY, you need frames
labelled with the real event that happened at that moment ("this was a pass," "this was
a shot") to check the guesser's answers against. That's ground truth, and:

- **SoccerNet's own datasets don't have it at the right granularity.** SoccerNet-GSR
  (the dataset the detector/tracker was trained on) has no action labels at all — only
  bounding boxes, tracks, and pitch calibration. SoccerNet's separate action-spotting
  dataset does label actions (pass, shot, etc.), but only as a rough single timestamp
  per action on the broadcast timeline, not tied to a specific player/track or a bounding
  box, and not fine-grained enough to validate a per-actor payload like ours (§3.5 of
  `PIPELINE_EVOLUTION.md`).
- **It is not otherwise publicly available pre-packaged for this.** No dataset combines
  SoccerNet-style player tracking with StatsBomb-style per-event, per-player labels on
  the same footage, as far as this review found.
- **We will have to curate it ourselves**, and the plan for doing so is exactly §4
  below (StatsBomb event data aligned to matching video) — a geometric baseline and a
  classifier both need the same (video moment → correct label) pairs, so building that
  alignment once serves both purposes. Until that alignment exists, the baseline can't
  be scored, only built.

A weaker, faster fallback: hand-label a small sample (a few hundred moments) ourselves by
eyeballing clips from the pipeline's own output — enough to sanity-check the baseline's
rough accuracy before committing to the larger StatsBomb-alignment effort below.

---

## 4. Training labels for the clip classifier (§7 item 5)

**Correcting the "thousands of clips" framing:** yes, realistically thousands, not
dozens. This is a standard fact about training small video classifiers from scratch —
a handful of examples per action type is not enough for a model to generalise across
different camera angles, lighting, player positions, and speeds. Sub-second visual
actions (a touch, a pass release) are also subtle and vary a lot frame-to-frame, which
tends to need more examples than a coarser problem, not fewer. There isn't a shortcut
around needing real volume here — it's why §7 item 5 calls out that SoccerNet's own
labels aren't dense enough, and why this doc treats getting labels as its own blocker.

**Why we don't have this today:**
- SoccerNet-GSR: no action labels (confirmed in §1 above).
- SoccerNet action-spotting: labels exist but are coarse (roughly one timestamp per
  action, not player-crop-level, not enough of the event types we need with the
  precision we need).

**Possible solutions, in order of effort:**

1. **Align StatsBomb open event data to matching broadcast video** (the plan already
   recorded in `PIPELINE_EVOLUTION.md` §7 item 5). StatsBomb's free/open-data
   competitions have precise, player-attributed event logs (pass, shot, carry, dribble,
   duel, etc., each with a timestamp, location, and outcome) — potentially thousands of
   labelled events per match, for free, without hand-labelling a single one. The catch:
   StatsBomb doesn't supply the video itself, so this only works for matches where we
   can separately obtain matching broadcast footage, and then have to solve the
   non-trivial problem of syncing StatsBomb's clock (minute/second within each half) to
   the actual video timeline (unknown kickoff offset, added time, broadcast cuts).
   **Not yet attempted or verified working — this needs a small proof-of-concept on one
   match before being trusted as the label source.**
2. **Hand-label our own clips.** Slower and more expensive per-label than option 1, but
   fully within our control and doesn't depend on finding matching video for a
   StatsBomb match. Realistic as a way to build an initial small validation/test set
   (to check option 1's alignment is correct) rather than as the primary source of
   thousands of training examples.
3. **Combine both:** bulk labels from StatsBomb-alignment for training volume, a smaller
   hand-labelled set held out as a trusted validation/test set that isn't subject to
   alignment errors.

None of steps 4/5 in `PIPELINE_EVOLUTION.md` §5 (the clip-classifier milestone) can
start in earnest until at least a proof-of-concept of option 1 (or a decision to go with
option 2 alone) exists.

### Feasibility check for option 1 (2026-09-21): the overlap exists

Matching StatsBomb's open-data matches against SoccerNet-v2's 550 games (bundled in the
`SoccerNet` pip package) by date plus both team names finds **133 games in both**:

| league / season | games |
|---|---|
| Premier League 2015/16 | 49 |
| La Liga 2015/16 | 34 |
| La Liga 2016/17 | 27 |
| Serie A 2015/16 | 9 |
| La Liga 2014/15 | 8 |
| Ligue 1 2015/16 | 3 |
| Bundesliga 2015/16 | 2 |
| La Liga 2019/20 | 1 |

That is thousands of labelled passes, carries and duels per match, so volume is not the
problem. The name matching is fuzzy (normalised tokens), so treat 133 as approximate and
eyeball each pair before use.

The clock-sync catch has a likely answer: SoccerNet-v2's action-spotting labels
(`Labels-v2.json`: kick-off, goals, shots, corners, cards... with a video timestamp per
half) cover the same games, so a kick-off and a goal per half give the offsets between
StatsBomb's per-period clock and the video. **Not yet done.** Still open:

1. The offset check itself: align kick-off and goal per half on one match, then confirm it
   on passes. This is the proof-of-concept the option asks for.
2. Raw broadcast video is separate from the GSR download (which needs no password); the
   SoccerNet video downloader needs the SoccerNet NDA password (the `video_ingestion` notebook
   that did this was removed 2026-10-12; it's in git history, commit `0bc1857`). SoccerNet-v2 video is
   224p or 720p rather than the 1080p GSR clips (from memory, verify), which matters for
   a ball that is already only 10–25px at 1080p.
3. Adopting StatsBomb's event definitions, as already planned.

---

## 5. Role and team classification for athlete detections (2026-09-21)

The athlete/ball model (§2) outputs no roles, but the pipeline needs
player / goalkeeper / referee and a team split. `gsr_pipeline/athlete_roles.py` restores
them, giving each detection one of: `player_team_a`, `player_team_b`, `goalkeeper_team_a`,
`goalkeeper_team_b`, `referee`, `other`.

**Technique** (no trained model; fitted per match):

1. **Colour.** Median Lab colour of the non-grass pixels in each box's upper body.
2. **Fit** on a bootstrap window: over-cluster the colours, merge clusters that are the
   same kit under different lighting, and take the two kits that appear at least 2.5 times
   per frame as the two teams.
3. **Predict.** Nearest fitted colour: a team kit means a player of that team. Any other
   colour within 20m of a goal line is the goalkeeper of the team defending that end; any
   other colour elsewhere is a referee; anyone well outside the pitch is `other`.

It needs each detection's pitch position, which the calibration stage already produces.

**Result** (`python -m gsr_pipeline.evaluate_athlete_roles`, `SNGS-021`, ground-truth boxes
and pitch positions so it isolates the classifier; fitted on the first 250 frames and
scored on the remaining 500):

| | held-out frames |
|---|---|
| role accuracy | 95.3% |
| team accuracy (players and keepers) | 97.6% |
| player recall | 97.3% |
| goalkeeper recall | 100% (79/79; no keeper is visible in the fit window) |
| referee recall | 74.3% |

**Limits, and what is not tested:**

- **One clip.** `merge_dist` was tuned here. Results were flat for 24–28, but 15–20 split
  one team's kit in sun and shade and mislabelled about 180 players as referees.
- **Referees are the weak class.** Their kit is sometimes close to a team's, and assistants
  stand on the touchline.
- **`other` is unvalidated.** It is position-only, and `SNGS-021` has no `other` examples.
  On-pitch staff are not caught.
- **A keeper or referee kit close to a team kit** is absorbed by that team.
- **`team_a` / `team_b` are arbitrary names** fixed at fit time. They need a flip at
  half-time (`team_a_defends_negative_x`), and binding them to home/away needs kit colours
  from match setup (see `TODO.md`).
- **Not wired into the pipeline**, and the fit is a batch step. The live path needs a
  bootstrap window plus periodic refits, like `LiveTeamAssigner`.

If it falls short on other matches, the upgrade is a supervised role classifier on crops:
GSR labels the roles (including `other`), so training data exists.

### Referee end-to-end breakdown for Path B (2026-09-23, corrected)

An earlier version of this section estimated end-to-end referee recall at ~67% by
multiplying the detector's referee recall (90.5%) by the classifier's referee recall in
isolation (74.3%, measured on ground-truth boxes). **That estimate was wrong** —
multiplying two stage-isolated numbers assumes their errors are independent, which they
are not (a box the detector localizes cleanly is also the box the classifier has an
easier time reading). It was never actually measured end-to-end, and should not have
been presented as one. Corrected: run the real detector on real frames, run the real
classifier on those real detections (real per-frame NBJW calibration for pitch
position, not ground-truth pitch position), then match the final labelled output
against ground truth.

**Real end-to-end run** (`yolo26s_athlete_ball` @640, real NBJW calibration per frame,
`athlete_roles.py` fitted on frames <250 and scored on the held-out rest, 150 frames
sampled every 5th frame across all 750 of `SNGS-021`):

| GT role | end-to-end recall (detected AND correctly labelled) |
|---|---|
| player | 99.2% (1298/1308) |
| goalkeeper | 100% (79/79) |
| referee | **88.1%** (104/118) |

1,505 of 1,585 held-out detections matched a real GT person (role accuracy among those:
98.4%; team-side accuracy for players+keepers: 98.8%). Of the 118 GT referees: 104
correct, 8 mislabelled goalkeeper (likely a referee standing near a goal-line zone), 6
mislabelled player (kit absorbed into a team-colour cluster), the remainder missed by
the detector entirely. 80 detections matched no GT person at all (59 labelled player, 19
referee, 2 other).

**88.1% is the number to use**, not 67% — referee end-to-end recall is meaningfully
below the person-level 97%+, but not as bad as the flawed estimate suggested. The
confusion breakdown above (goalkeeper/player absorption) is the concrete lead for
improving it further, not a generic "referees are hard" framing.
