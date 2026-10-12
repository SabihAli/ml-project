# CV Service — Event Detection Architecture

Status: **partially implemented** (last reviewed 2026-09-13). This document records the
decisions for evolving `ml-project` (the existing GSR pipeline) into a full CV service
that emits structured match events to `data_ingestion`.

| Area | Status |
|---|---|
| §1 deployment model | **Changed**: cv-service runs as its own service behind HTTP + WS; `CVPipelineSource` is a network client, not a library import (see §1). |
| §3.1 rule-based tier | `PLAYER_POSITION`, `BALL_POSITION`, `REFEREE_POSITION`, `PLAYER_IDENTIFIED` emitted live. `POSSESSION`/`TURNOVER` exist only in the offline batch path (`possession.py`), not the live stream. |
| §3.2 classifier tier | Not started. Gated on §3.4 (revision semantics) and the risks in §7. |
| §4 Kafka mapping | Done for every v1 type, including `REFEREE_POSITION`. |
| §5 build order | Steps 1 and 3 done; step 2 partial (batch only); steps 4-5 not started. |

`docs/architecture/STREAMING_ARCHITECTURE.md` §8 is the authoritative record of what has been built
and verified. This document is authoritative for the event taxonomy (§3) and the
revision semantics (§3.4).

---

## 1. Scope of the CV service

The CV service owns the entire **video → structured event** surface. It is the only
component in BallKnowledge with access to raw broadcast video/frames — everything
downstream (`feature_service`, etc.) consumes structured JSON/event data only, never
pixels. This means any event that requires visual context (cropping around a player,
watching a sequence of frames) must be computed here; it cannot be pushed to a
downstream service that only sees structured data.

**Output contract:** the CV service does **not** write JSON files as its output in
any deployment. Its events reach `data_ingestion`, which is the existing
data-plumbing layer for the whole system (see [`data_ingestion/README.md`](../../data_ingestion/README.md),
[`data_ingestion/kafka.md`](../../data_ingestion/kafka.md)). `data_ingestion` owns
Kafka publishing (`KafkaEventPublisher`, per-group topics, backpressure
handling) — the CV service never touches Kafka directly.

### Deployment model (decision, revised)

**Current decision: a network boundary at cv-service's own API.**

- cv-service runs as a separate process exposing `POST /streams`,
  `POST /streams/{id}/stop`, `GET /health`, and a WebSocket event feed
  (`cv_service/api/`, `docs/architecture/STREAMING_ARCHITECTURE.md` §3-4).
- `CVPipelineSource` (`data_ingestion/src/data_ingestion/cv_pipeline_source.py`) is a
  `BaseDataSource` that registers a stream over HTTP, consumes the WS feed on a
  background thread, and translates cv-service's raw "fast"/"appearance" messages into
  `DataEvent`s. It does not import `gsr_pipeline`.
- cv-service's GPU-heavy dependencies (torch, ultralytics, easyocr) therefore never
  enter `data_ingestion`'s environment, and cv-service can move to its own pod without
  becoming a Kafka producer itself — `data_ingestion` keeps sole ownership of Kafka.

**Superseded original decision (kept for context):** the first version of this plan
made `CVPipelineSource` import `gsr_pipeline` in-process, and deferred any pod split
because `DataSourceRegistry.subscribe()` stores a Python callable that cannot cross a
process boundary — a split would have required cv-service to become its own Kafka
producer. Putting the boundary at cv-service's API instead avoids that refactor
entirely, which is why the in-process plan was dropped.

**Known cost of the boundary:** the WS feed has no replay buffer (drop-oldest
backpressure, nothing retained across a reconnect — `STREAMING_ARCHITECTURE.md` §4.3).
Events lost between cv-service and `CVPipelineSource` are lost before Kafka sees them,
so the at-least-once guarantee in §4 applies from `data_ingestion` onwards only.

---

## 2. Latency architecture

Ball/referee-event detection (the classifier-based tier, see below) requires a
sliding window of frames before it can classify anything — that latency is
irreducible, only tunable via window size. What **is** avoidable is *additional*
latency from being gated behind pipeline stages it doesn't need.

- Event detection consumes output directly off **detection + tracking**, not
  after team-classification / jersey-OCR / pitch-calibration / tracklet-refinement.
  Those stages don't block it.
- This is a **parallel branch off tracking**, not a strictly sequential pipeline.
  Implemented for the live path as `process_stream`'s fast/appearance split: the fast
  loop emits position events every frame while ReID, jersey OCR, and team assignment
  run on `ReidWorkerPool` and arrive later as appearance events
  (`STREAMING_ARCHITECTURE.md` §1.7.6).
- **Divergence to resolve:** the implemented fast path is detect → role → track →
  **calibrate** (`_detect_track_calibrate`). Calibration is therefore on the fast path,
  contrary to the first bullet. Pitch coordinates are needed for position events, so
  this may be correct for them; it should not also gate the classifier tier. Measure
  per-stage latency before building step 4 and decide explicitly (§7).
- Window size strategy: **run both a short window (~0.5–1s / 12–25 frames) and a
  longer window (~2–3s) in parallel.** Short window fires fast for discrete/instant
  actions (touch, shot release); long window has more context for actions that need
  before/after disambiguation (tackle vs. turnover vs. carry) and arrives later as a
  revision of the fast prediction, under the semantics in §3.4. Exact frame counts for
  each window are **not yet decided** — to be set empirically after looking at real
  extracted clips, not guessed in advance.

---

## 3. Event taxonomy (v1)

15 event types total: 6 rule-based, 9 classifier-based. All are additions to (or
reuses of) the existing `EventType` enum in
`data_ingestion/src/data_ingestion/custom_types.py`.

### 3.1 Rule-based (from tracking only, no clip model — ship first)

These require no ML classifier, only per-frame track + ball position data that
`gsr_pipeline` already produces. Every detector role maps to exactly one position
event type; no role may fall through to another role's type.

| Event | Status | Detection method |
|---|---|---|
| `PLAYER_POSITION` | emitted live | Players **and goalkeepers**, per frame. |
| `BALL_POSITION` | emitted live | Ball detections only, per frame. |
| `REFEREE_POSITION` | emitted live (added 2026-09-13) | Match officials, per frame. Separate from `PLAYER_POSITION` so consumers never count officials as players, and kept (not dropped) because the deferred referee-event milestone (§3.3) needs officials' positions. See the note below. |
| `PLAYER_IDENTIFIED` | emitted live | Jersey number, role, team. Emitted independently of position — on first confirmation or on change, **not every frame** (avoids coupling identity updates to position-event frequency). |
| `POSSESSION` | batch only | Which track_id/team currently has the ball. Rule-based (proximity ellipse + hold-frame grace window, `possession.py`), computed by the CV service and sent to `data_ingestion` like any other event — **not** merely internal plumbing. Emission policy is open, see §7. |
| `TURNOVER` | batch only | Possession changes from one team to the other. Derived directly from the `POSSESSION` signal — rule-based comparison of current vs. previous possessor's team. Whether cv-service should emit it at all is open, see §7. |

**Why `REFEREE_POSITION` exists.** The first version of this table listed only
`PLAYER_POSITION` and `BALL_POSITION`, saying nothing about referees, even though the
detector classifies them directly (`detect.py`: player=0, goalkeeper=1, referee=2,
ball=3). `CVPipelineSource` filled the gap with a person-vs-everything-else split, so
**every referee detection was published as `BALL_POSITION`** — typically 1–3 phantom
balls per frame on the `positional` topic, corrupting anything that reads ball
position. Fixed by mapping every role explicitly and skipping (with a warning) any
unrecognised role instead of defaulting it; regression-tested in
`data_ingestion/tests/test_cv_pipeline_source.py`.

Note: an earlier draft of this taxonomy proposed a separate `POSSESSION_CHANGE`
event distinct from `TURNOVER`. That was rejected as redundant — `TURNOVER` covers
the "possession changed team" signal on its own.

### 3.2 Classifier-based (needs the cropped-clip sequential model)

These require visual motion context (a short cropped clip around the ball
possessor, or the referee, fed through a 3D CNN / ConvLSTM / other sequential
video model). The claim that they cannot be reliably inferred from position/geometry
alone is **untested** — see §7.

| Event | Status in enum | Notes |
|---|---|---|
| `PASS` | existing | |
| `SHOT` | existing | **One type with an `outcome` field** (decided 2026-09-14), not one type per outcome. Outcome values copy StatsBomb's shot outcomes: `blocked`, `goal`, `off_target`, `post`, `saved`, `saved_off_target`, `saved_to_post`, `wayward` (confirm the list against the StatsBomb open-data spec when implementing). The short window emits `SHOT` with `outcome: null`; the long window fills in the outcome as a revision (§3.4). |
| `CLEARANCE` | existing | **Moved into v1** (2026-09-14). Visually a kick; told apart from a long pass by context (defensive third, pressure, no intended recipient). StatsBomb labels it. |
| `CARRY` | existing | Possessor moves with the ball, **uncontested** (not being challenged). |
| `DRIBBLE` | existing | Possessor moves with the ball and **beats a defender** — distinct from `CARRY`. Definition under review, see §7. |
| `TACKLE` | **new** | Not previously in the enum. |
| `TOUCH` | **new** | Renamed from an earlier "first touch" concept. Explicitly rejected as rule-based — a simple "ball entered proximity radius" heuristic would produce false positives (ball merely passing near a player) and miss true touches (controlled touch with minimal ball displacement); requires visual context to know contact actually occurred. Needed to accurately count touches per possession. **Not a single-frame problem either:** at 25 fps frames are 40ms apart and foot–ball contact lasts ~10ms, so usually no frame shows the contact; the evidence is the ball changing direction/speed across frames. Needs a short window of ~3–5 frames (a small 2D CNN over stacked frames is enough), not the 0.5–1s model. Limited directly by ball detection (§6). |
| `INTERCEPTION` | existing | **Redefined**: a sub-type of `TURNOVER`, specifically for when the classifier confirms a pass was read/cut out (judged on pass quality/intent), not just "ball touched by an opponent mid-flight." The intercepting player does not need to retain control for this to count. Emitted after the related `TURNOVER`, once the clip resolves, as a *related* event under §3.4 (not a revision of the `TURNOVER`). |
| `DUEL` | existing | |

### 3.3 Explicitly deferred (not in v1 scope)

- **Restarts and stoppages** — corner, goal kick, throw-in, free kick, penalty,
  kick-off, goal (decided 2026-09-14):
  1. `BALL_OUT_OF_PLAY` is detected from tracking (ball leaves the pitch).
  2. The restart **type** is confirmed by a whole-frame classifier when the restart
     is taken — these scenes are distinctive and last seconds — and published as a
     revision (§3.4). `feature_service`'s restart detector already accepts an
     explicit `out_type` and trusts it fully, so the classifier plugs in there.
  3. `GOAL`: combine `SHOT` with `outcome: goal`, the ball crossing the goal line,
     and the score changing on the broadcast scoreboard.
  Cards and fouls stay a later milestone using referee crops.
- Existing-but-unused `EventType` values not part of this taxonomy:
  `GOALKEEPER`, `PRESSURE`, `BALL_OUT_OF_PLAY`, `FOUL_COMMITTED` (the last two are
  part of the deferred referee-event work). `DRIBBLE` vs `CARRY` was flagged as a
  possible overlap and resolved — both are distinct and both are in scope (see
  table above).

### 3.4 Revision semantics (decided 2026-09-13 — prerequisite for step 4)

The dual-window design (§2) and `INTERCEPTION` (§3.2) both publish something that
changes the meaning of an earlier event. `DataEvent` has no field for that today, and
Kafka delivery is at-least-once, so without an explicit protocol a consumer cannot tell
a correction from a duplicate or a new event. **No classifier-tier event may be emitted
until this is implemented.** Nothing emits revisions today, so this is a recorded
decision, not code.

**Two distinct cases:**

1. **Revision** — a later, better answer about *the same moment*. Example: the short
   window says `SHOT`; the long window, with more context, says `PASS`.
2. **Related event** — a *new fact* that depends on an earlier one. Example:
   `INTERCEPTION` explains the cause of an already-published `TURNOVER`; the
   `TURNOVER` itself remains true.

**Fields** (carried in `DataEvent.metadata`, so no envelope change; promote to
first-class `DataEvent` fields if more than one consumer relies on them):

| Key | Type | Meaning |
|---|---|---|
| `event_uid` | str | Stable ID of one logical event, assigned by the producer at first emission. Scoped to the moment, **not** the event type: `{source_id}:{anchor_frame_idx}:{anchor_track_id}`, where the anchor is the ball-contact frame and possessor track the classifier was run for. |
| `revision` | int | `0` on first emission, incremented by 1 per revision of the same `event_uid`. |
| `final` | bool | `true` on the last revision the producer will ever send for this `event_uid` (normally the long-window result). |
| `status` | str | `"active"` (default) or `"retracted"` — the moment turned out not to be an event at all. |
| `related_event_uid` | str, optional | Only on related events: the `event_uid` this event explains or depends on. |

**Rules:**

- **Revision, including a change of event type.** Re-publish under the same
  `event_uid` with `revision + 1`, carrying the new `event_type` and payload. A
  contradiction (`SHOT` → `PASS`) is an ordinary revision, not a special case.
- **Retraction.** Re-publish the same `event_uid` with `revision + 1`,
  `status="retracted"`, `final=true`, and the last `event_type` unchanged.
- **Consumer resolution.** Per `event_uid`, the highest `revision` wins. Receiving a
  `(event_uid, revision)` already seen is a duplicate and is ignored, which makes
  processing idempotent under at-least-once delivery. A lower revision arriving after
  a higher one is ignored.
- **Consumers that cannot revise** (anything with irreversible side effects) act only
  on `final=true`, trading latency for certainty.
- **Ordering constraint.** Every revision of one `event_uid` must stay in one Kafka
  topic group, so per-`match_id` partition ordering still applies to it. All §3.2
  types are in `ball`; a revision may never change an event into a type in another
  group.
- **Related events** get their own `event_uid` and set `related_event_uid`. They never
  modify the event they reference.
- **Rule-based events** (§3.1) are emitted once and never revised; they omit these
  keys, and consumers treat a missing `event_uid` as a final, non-revisable event.

### 3.5 Payloads (draft, 2026-09-14)

**Coordinates: 0–1 relative to the pitch** (decided). `x` runs along the pitch length,
`y` along the width, both in [0, 1]. cv-service can produce these directly: calibration
already maps every frame onto a fixed 105×68 template, so
`x = (x_m + 52.5) / 105` and `y = (y_m + 34) / 68`. The coach-entered pitch size
(`length_m`/`width_m` in match setup) is only needed downstream to turn 0–1 back into
metres for distances and speeds. Caveat: the template assumes standard markings, so on
a non-standard pitch values near the penalty areas are slightly off. Which goal is
`x = 0` and which touchline is `y = 0` must be confirmed against real footage.

**Identity** (decided): every person reference is an `actor`:

```json
{"track_id": 7, "team_side": "left", "jersey_number": "10", "player_id": null}
```

`track_id` is always present (except where noted). `team_side` and `jersey_number` are
filled when known. `player_id` stays `null` in this phase — resolving it (jersey number
→ coach lineup) is later work (§7).

**Fields on every on-ball event:** `actor`, `location {x, y}` at the event frame,
`frame_idx`, `timestamp_ms` (frame time, not send time), `confidence`,
`window` (`"short"` | `"long"`), and the §3.4 revision keys.

| Event | Extra payload fields |
|---|---|
| `PASS` | `recipient` (actor, nullable), `end_location` (nullable), `outcome`: `complete` / `incomplete` / `out` / null |
| `SHOT` | `actor` is the shooter and **may be null** if the clip doesn't show them (recoverable from the possession timeline). `outcome` (see §3.2, null until the long window), `end_location` (nullable), `blocker` (actor, nullable, for `blocked`), `goalkeeper` (actor, nullable, for saves) |
| `CARRY` | `start_location`, `end_location`, `duration_ms` |
| `DRIBBLE` | `opponent` (actor, nullable), `outcome`: `complete` / `incomplete` |
| `TOUCH` | none |
| `CLEARANCE` | `end_location` (nullable) |
| `TACKLE` | `opponent` (actor, nullable) — the player tackled; `outcome`: `won` / `lost` |
| `DUEL` | `player_a`, `player_b` (actors), `winner`: `a` / `b` / null, `duel_type`: `ground` / `aerial` / null (aerial can't be inferred from position alone — no ball height) |
| `INTERCEPTION` | `related_event_uid` pointing at the `TURNOVER` it explains (§3.4) |

---

## 4. Kafka / data_ingestion integration notes

- Every `EventType` value must be listed in `TOPIC_GROUPS` in
  `data_ingestion/src/data_ingestion/kafka_publisher.py` — that module **raises at
  import** if any enum value is left unmapped, so this is a hard requirement, not
  a nice-to-have.
- Group assignment (implemented):
  - `positional`: `PLAYER_POSITION`, `BALL_POSITION`, `REFEREE_POSITION`,
    `PLAYER_IDENTIFIED`.
  - `ball`: `POSSESSION`, `TURNOVER`, `TACKLE`, `TOUCH`, alongside the existing
    ball-action types (`pass`, `shot`, `dribble`, `clearance`, `interception`, `carry`,
    `duel`, `goalkeeper`).
- The CV service's events use the envelope and routing already documented in
  `data_ingestion/kafka.md` (`match_id` partition key, at-least-once delivery,
  `schema_version` field). Two corrections to what the first version of this
  section assumed:
  - **Partition order is arrival order, not event-time order.** Appearance events
    arrive after, and out of order with, fast events for later frames, and
    `CVPipelineSource` stamps `PLAYER_IDENTIFIED` with wall-clock time at emission
    rather than frame time. `timestamp_ms` is therefore not monotonic across event
    types; consumers joining on time must not assume it is.
  - **At-least-once starts at `data_ingestion`**, not at cv-service (see §1).

---

## 5. Build order

1. ✅ Add new `EventType` enum values (`PLAYER_IDENTIFIED`, `POSSESSION`, `TURNOVER`,
   `TACKLE`, `TOUCH`, and later `REFEREE_POSITION`) to `custom_types.py` and map them
   into `TOPIC_GROUPS`.
2. ◐ Possession assignment + `TURNOVER` rule-based logic — pure logic module,
   consuming per-frame track + ball positions, no video access needed. Exists as
   `gsr_pipeline/possession.py`, but only over a completed sequence in the batch path;
   not yet computed incrementally inside `process_stream`.
3. ✅ `CVPipelineSource` (`BaseDataSource` subclass) in `data_ingestion/`, emitting
   `PLAYER_POSITION`, `BALL_POSITION`, `REFEREE_POSITION`, and `PLAYER_IDENTIFIED` as
   `DataEvent`s in real time — via cv-service's WS feed (§1), not an in-process import.
   `POSSESSION`/`TURNOVER` are not emitted until step 2 runs live.
4. Clip-classifier work (crop extraction around the ball possessor, dual
   short/long window extraction, model architecture for the 8 classifier-based
   event types) — its own milestone, after step 3 is working end-to-end.
   **Prerequisites:** §3.4 implemented, and the §7 risks addressed (ball-detection
   quality, labels, geometric baseline).
5. Referee-crop classifier (foul/card/corner/goal-kick) — deferred milestone,
   reusing the clip-classifier infra from step 4 as a second head/dataset.

---

## 6. Current state

Pipeline quality, as measured:

- `gsr_pipeline` runs end-to-end: YOLOv8n detection → OSNet ReID tracking → KMeans
  team clustering → EasyOCR jersey recognition → NBJW pitch calibration → tracklet
  refinement → annotated video + `predictions.json`. Verified on the `SNGS-021` sample
  sequence (75 frames): DetA 50.48%, AssA 95.43%, GS-HOTA 69.10% (kickoff segment, not
  representative of full-match difficulty).
- **Per-class detection on `SNGS-021`** (all 750 frames, `gsr_pipeline/evaluate_detection.py`,
  config thresholds; people matched at IoU ≥ 0.5, ball at centre distance ≤ 15px):

  | Class | GT | Recall | Precision |
  |---|---|---|---|
  | player | 10,338 | 95.4% | 95.5% |
  | goalkeeper | 397 | 99.5% | 99.5% |
  | referee | 1,191 | 84.6% | 81.8% |
  | **ball** | **752** | **52.4%** | **76.7%** |

  Ball detail: found in 52.5% of the 750 ball-labelled frames; 40 frames contain more
  than one ball detection; the longest unbroken miss is **168 frames (6.7s)**. Recall
  is 31.7% for balls 10–20px across and 63.7% for balls ≥ 20px.

  Referee detail, a detector defect in its own right: **224 of the 1,232 referee
  detections (18%) are wrong** — 40 are labelled players, and 184 match no labelled
  person at IoU ≥ 0.5 (duplicates, loose boxes on real referees, or unlabelled people;
  not yet broken down). A further 183 real referees are missed. So roughly 1 in 5
  `REFEREE_POSITION` events is not a correctly located referee, and the 40 players are
  also missing from `PLAYER_POSITION`. Before the 2026-09-13 routing fix these same
  detections were published as `BALL_POSITION`.

  Caveats: one 30-second kickoff clip, not a benchmark. The detector was never trained
  on `SNGS-021`, but its checkpoint was selected on an unseeded random 65% of the
  `valid` split, which may include it, so these numbers may be slightly optimistic.
  Re-run: `python -m gsr_pipeline.evaluate_detection --sequence_dir <seq>`.
- Fixed a real bug in `gsr_pipeline/track.py`: `BYTETracker(args, frame_rate=...)`
  broke against `ultralytics==8.4.114` (constructor dropped the `frame_rate` kwarg
  in newer versions). Patched with a `TypeError` fallback.
- `data_ingestion` also has a **batch/offline** GSR integration:
  `data_ingestion/scripts/build_golden_fixture_from_gsr.py` reads
  `Labels-GameState.json` / pitch-coordinate output and converts it into
  `player_position` `DataEvent`-shaped JSON for test fixtures. Useful as a reference
  for the position → payload mapping (team_id/player_id scheme, coordinate space
  conversion); it is separate from the live path in §1.

For the build status of the live streaming path, see `STREAMING_ARCHITECTURE.md` §8.

---

## 7. Open issues from the 2026-09-13 review

Recorded, not decided. Each needs an explicit call before the step it affects.

1. **Ball-detection quality gates everything ball-based.** Possession, touch, pass,
   and shot all depend on ball position. Measured in §6: the ball is missed in about
   half of all frames and gaps run up to 6.7s, which on its own makes live possession
   unreliable. The ball occupies ~10–25px in a 1080p frame, and the detector runs at its training size of
   `imgsz=640` (about 3× downscaling). Options: higher inference resolution (retrained
   at that size), tiled/sliced inference around the last known ball position, or a
   dedicated ball detector/tracker. Resolve before step 2 goes live or step 4 begins.
   **Must be resolved before proceeding further — see `docs/path_b/DETECTOR_AND_DATA_BLOCKERS.md`
   §1-2 for what the detector is, its fine-tune settings, and the improvement options.**
2. **`POSSESSION` emission policy.** Emitted per frame it is ~25 mostly-unchanged
   events/second. Edge-triggered (on change, like `PLAYER_IDENTIFIED`) is likely
   right, but consumers that want a level signal must then hold state.
3. **Should cv-service emit `TURNOVER` at all?** It is a pure function of
   `POSSESSION`. Deriving it in `feature_service` would keep it replayable and fixable
   without touching the CV pipeline. The same argument applies to `INTERCEPTION` as a
   `TURNOVER` sub-type.
4. **Geometric baseline before the clip classifier.** The claim that `PASS`/`SHOT`/
   `CARRY` need visual context is untested. A trajectory-and-possession baseline is
   cheap, sets the bar the classifier must beat, and can pre-label clips for it.
   **Must be resolved before proceeding further — see `docs/path_b/DETECTOR_AND_DATA_BLOCKERS.md`
   §3: there is no ground truth to score this baseline against today, and building one
   is the same work as item 5's label-sourcing problem below.**
5. **Training labels for §3.2.** SoccerNet-GSR has no action labels. SoccerNet's
   action-spotting datasets are coarse and frame-level rather than player-crop-level.
   Aligning StatsBomb event data (`data_ingestion/statsbomb_source.py`) to matching
   video is the most plausible source of labels at scale, which only works if §3.2
   adopts StatsBomb's event definitions rather than its own.
   **Must be resolved before proceeding further — see `docs/path_b/DETECTOR_AND_DATA_BLOCKERS.md`
   §4 for why this realistically needs thousands of labelled clips and the possible
   ways to source them.**
6. **`DRIBBLE` definition.** "Beats a defender" is an outcome judged over seconds and
   is not observable in a 0.5–1s clip. StatsBomb defines a dribble as an *attempt*
   with a complete/incomplete outcome; aligning with that also serves issue 5.
7. **Calibration on the fast path** (§2): measure its per-frame cost and decide.
8. **No match clock.** Nothing maps stream time to match time or emits
   `MATCH_START`/`HALF_START`, so CV events cannot be joined to other sources by time.
9. **No player identity.** `PLAYER_IDENTIFIED` carries a pitch side (`"left"`/
   `"right"`), not a team, and a jersey number, not a player, keyed on a `track_id`
   that changes after occlusions. Nothing maps these to a roster `player_id`.
   **Intended design vs. code (verified 2026-09-15):** the design says `player_id` is
   unresolved at setup and filled in by CV — `feature_service/docs/FEATURE_SCHEMA_REFACTOR.md`
   marks `SquadInfo.player_id` "CV-resolved", and `data_service` has a nullable
   `entity_id` "filled by identity resolution" plus
   `PATCH /matches/{id}/teams/{team}/squad/{jersey}/identity`. None of it is wired:
   `feature_service`'s match setup **requires** `player_id` per player and the column
   is `NOT NULL`; nothing calls `data_service`'s identity endpoint; no service consumes
   `PLAYER_IDENTIFIED`. The only other write of `player_id` is substitutions. The
   intended route stays: OCR jersey → match to the coach's lineup by team + jersey →
   write `player_id`. Jersey OCR is precise but rarely succeeds — measured on
   SNGS-021: number not visible in 52% of player boxes; of visible numbers, 12.9% read,
   86% of reads correct; 5 of 9 visible-number tracks resolved, none wrong. Deferred
   past this phase.
10. **`feature_service` ignores every CV position event today.** Its position
    consumers (formation, press, transition) read `team_id`, `player_id`, `x`, `y` and
    skip events without `team_id`/`player_id`. `CVPipelineSource` sends `track_id`,
    `role`, `pitch_x`, `pitch_y`. Fix when identity work happens: send `x`/`y` in 0–1,
    map `team_side` → `team_id` using the match's attacking direction, and look up
    `player_id` from the lineup — or let consumers key on `track_id` meanwhile.
11. **`feature_service` pitch format — fixed 2026-09-15.** Press, transition detection
    and set-piece recognition hardcoded StatsBomb's 120×80. They now take a
    `helpers.pitch.Pitch` built from the coach-entered `length_m`/`width_m`, and expect
    positions in metres from a corner of that pitch (the convention formation already
    used). Zones scale with the pitch; the penalty area uses its fixed Laws size
    (16.5m × 40.32m). Mock demos pass `STATSBOMB_PITCH` explicitly because their
    generators emit 120×80. Still open: moving everything to 0–1 is a later
    conversion at ingestion, and the StatsBomb source must convert its 120×80
    coordinates before feeding these modules. Found while fixing: the free-kick
    classifier marked every foul outside the penalty areas indirect (fixed).
12. **Calibration assumes a 105×68 pitch.** `gsr_pipeline/nbjw_calib/utils/utils_calib.py`
    hardcodes the pitch template at 105×68. Markings the Laws fix (penalty area, goal
    area, centre circle) are right on any pitch, but touchline, goal-line and corner
    keypoints move with pitch size, so any frame fitted from them is wrong on a
    non-standard pitch (up to ~2.5m at each end on 100×64; the Laws allow 90–120m ×
    45–90m). SoccerNet's labels also assume 105×68, so evaluation cannot detect this.
    Fix: build the template from the match's `length_m`/`width_m` (law-fixed markings
    stay fixed, outer lines move) and pass pitch size in `POST /streams`. The code is
    vendored third-party (edit deliberately, per AGENTS.md).
13. **Bad projections aren't filtered.** Pipeline output on SNGS-021 includes
    `y = −71.8m`, 38m beyond the touchline, from a bad homography. Positions far
    outside the pitch should be dropped or flagged.
14. **Restart detector problems** (`feature_service/set_piece_recognition/restart_detector.py`,
    no tests beyond an import check):
    - Gets no real data: the StatsBomb source maps "Foul Committed" to `PRESSURE` and
      never emits `BALL_OUT_OF_PLAY`; the replay demo sends `location: [x, y]` but the
      detector reads `payload["x"]`, so it fails with a swallowed `KeyError`.
    - Attacking direction is read once and never flipped at half time
      (`set_piece_handlers.py` doesn't subscribe to `HALF_START`), so second-half
      corners and goal kicks come out swapped. `event.half` doesn't exist on
      `DataEvent`, so features are always stamped first half.
    - "Last touch" only updates on `PASS`/`CARRY`; clearances, blocked shots,
      deflections and saves — the usual causes of corners — are ignored, so most
      corners are lost.
    - A ball out within 6m of a goal line counts as crossing the goal line, so
      throw-ins next to the corner flag become corners or goal kicks.
    - Unknown last touch awards the throw-in to whichever team is first in the list.
    - "Home" is the first database row; `AttackingDir.LEFT`'s meaning is undocumented.
    - Goal kicks recognised but never emitted; no kick-off, drop ball, or offside free
      kick.
    - Minute/second assume one continuous clock; `timestamp` is `datetime.now()`.
