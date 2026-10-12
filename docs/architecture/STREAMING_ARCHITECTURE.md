# CV Service — Live Streaming Architecture

Status: **implemented for the Path A live path** (2026-10-12 review). §8 is the
authoritative per-step build status. Path B's live pipeline is planned separately in
`docs/path_b/plans/IMPLEMENTATION_PLAN.md`. This document is the detailed design for
two decisions made in discussion (see chat history referenced in `docs/architecture/REQUIREMENTS.md`
§5) and scopes them to an implementable level before any code is written:

1. **Frame-source abstraction** inside `cv_service`, so the pipeline consumes frames
   the same way regardless of whether they come from a pre-extracted image directory,
   a local video file, or a live broadcast stream.
2. **Network architecture** between `cv_service` (its own process/service, going
   forward) and `data_ingestion`'s `CVPipelineSource` — WebSocket-based, superseding
   `PIPELINE_EVOLUTION.md`'s original in-process/library-import plan.

This document does not replace `PIPELINE_EVOLUTION.md` — the event taxonomy, latency
architecture (dual short/long window), and Kafka topic-group mapping decided there
still stand. This document covers *how frames get into the pipeline* and *how events
get out of cv-service*, which `PIPELINE_EVOLUTION.md` left as an in-process assumption
that this document now revises.

Framework decision: **FastAPI**, for the HTTP + WebSocket endpoints in §3.

---

## 1. `FrameSource` abstraction

### 1.1 Interface

New module: `gsr_pipeline/frame_source.py`.

```python
@dataclass
class FrameRecord:
    frame_idx: int          # 1-based, monotonically increasing within a source
    image: np.ndarray       # BGR, as read by OpenCV
    timestamp_ms: int       # source-relative; see per-implementation notes below

@dataclass
class ValidationResult:
    ok: bool
    failure_reason: Optional[str]   # one of the categories in §1.4, or None if ok
    detail: str = ""                # human-readable, for logs/response body

class FrameSource(ABC):
    def connect(self) -> None: ...
    def read_frame(self) -> Optional[FrameRecord]: ...   # None = exhausted/ended
    def close(self) -> None: ...
    def validate(self, max_frames: int, timeout_s: float) -> ValidationResult: ...
```

`validate()` has a default implementation on the ABC (connect, read up to
`max_frames` frames or until `timeout_s` elapses, run the black/frozen check from
§1.4, close) — subclasses only override it if a source type needs a different check
(none currently do).

### 1.2 Implementations

| Class | Backing | Notes |
|---|---|---|
| `ImageDirFrameSource` | `utils.load_frames_from_dir` (unchanged) | Wraps today's behavior exactly — used for the existing offline SoccerNet benchmark path. `timestamp_ms` is synthesized from a configured nominal fps (no real timestamps in the dataset). |
| `VideoFileFrameSource` | `cv2.VideoCapture(path)` | Used both for real local-file input and, during development, as a stand-in for "live" mode without needing an actual broadcast feed (see §5). `timestamp_ms` from `cap.get(cv2.CAP_PROP_POS_MSEC)`. |
| `LiveBroadcastFrameSource` | `cv2.VideoCapture(url)` (OpenCV's FFmpeg backend handles RTMP/RTSP/HLS URLs transparently) | The production input. `read_frame()` returning `None` here means the connection dropped, not "sequence complete" — callers must distinguish this from `ImageDirFrameSource`/`VideoFileFrameSource` exhaustion (see §1.3 state semantics). |

Constructor args are source-specific (`ImageDirFrameSource(path)`, `VideoFileFrameSource(path)`,
`LiveBroadcastFrameSource(url, reconnect_policy=...)`) — no shared config schema is
imposed beyond the common `FrameSource` interface.

### 1.3 `None` semantics differ by source type — must not be conflated

- `ImageDirFrameSource` / `VideoFileFrameSource`: `read_frame() -> None` means the
  sequence is complete. This is a normal, expected end state.
- `LiveBroadcastFrameSource`: `read_frame() -> None` means the connection was lost
  or the source stopped sending frames unexpectedly. This is **not** a normal end
  state for a live stream (a live match doesn't "run out of frames" the way a file
  does) — a caller driving `process_stream` (§2) must treat this as a
  disconnect/error condition, not silently finish. `LiveBroadcastFrameSource` is
  responsible for surfacing this distinction (e.g. a `.disconnect_reason` property
  set before returning `None`), not the caller guessing from source type.

### 1.4 Validation checks (FR-002)

Implements the "thoroughly test the stream source first" requirement from
`docs/architecture/REQUIREMENTS.md` FR-002. Checks, in order, each mapping to a distinguishable
`failure_reason` (final HTTP status-code mapping happens in the API layer, §3.3 —
this layer only classifies, it doesn't know about HTTP):

1. **`unreachable`** — `connect()` raises or times out (source not reachable / DNS
   failure / connection refused).
2. **`undecodable`** — connected, but no frame could be decoded within `max_frames`
   attempts (unsupported codec/container, corrupt stream).
3. **`no_usable_video`** — frames decode, but the content is unusable:
   - **black/frozen check**: compute mean pixel intensity and frame-to-frame
     absolute difference across the sampled frames. Reject if mean intensity is
     below a configured floor (near-black) for all sampled frames, **or** if
     consecutive-frame difference stays below a configured floor for all sampled
     frames (frozen/static image passed off as a stream).
   - Both thresholds are config values (`possession.py`'s pattern of "no hardcoded
     tunables" applies here too — see §6 config additions), not hardcoded, since
     the right sensitivity is expected to need empirical tuning against real
     broadcast feeds, not guessed here.
4. Success — `ValidationResult(ok=True)`.

**Not decided here (explicitly out of this document's scope):** the exact numeric
thresholds for "near-black" and "frozen" — these need tuning against real sample
footage, which doesn't exist yet in this repo. Config keys are defined with
placeholder defaults; treat those defaults as provisional.

### 1.5 `pipeline.py` changes

`GSRPipeline.process_sequence` currently calls `utils.load_frames_from_dir` directly
and holds the full frame list in memory before the loop starts. It is refactored to
accept a `FrameSource` instead:

```python
def process_sequence(self, frame_source: FrameSource, max_frames: Optional[int] = None) -> dict:
    ...
```

For `ImageDirFrameSource`, behavior is unchanged (still reads everything up front,
since batch/Phase 2 needs the complete sequence regardless). A thin compatibility
wrapper keeps `run.py --sequence_dir` working unchanged (constructs an
`ImageDirFrameSource` internally) — no CLI-facing behavior change, this is purely an
internal refactor. **Verification requirement**: run `gsr_pipeline.run` against the
`valid_sample` sequence (then in the repo) before and after this change and confirm identical
`predictions.json` output (byte-for-byte on the deterministic fields) — this is a
refactor, not a new capability, and must not change existing behavior.

### 1.6 `process_stream` — the live-mode counterpart

New method, `GSRPipeline.process_stream(frame_source, on_frame_result, stop_event)`:

- Loops `frame_source.read_frame()` until it returns `None` or `stop_event` is set.
- Per frame, runs **Phase 1 only** (detect → reid/role → track → calibrate → jersey)
  — the same stages `PIPELINE_EVOLUTION.md` §2 already scoped as the latency-critical
  path, decoupled from team/refine.
- Calls `on_frame_result(frame_result)` per frame instead of accumulating a `full_df`
  — there is no Phase 2 in streaming mode yet (see decision below).
- On a `LiveBroadcastFrameSource` disconnect (§1.3), stops the loop and reports the
  disconnect reason to the caller rather than treating it as normal completion.

**Team assignment in live mode — decided: NOT deferred.** Superseded by §1.7 below
(a full design, not a stub). `process_stream` emits position (`PLAYER_POSITION`,
`BALL_POSITION`), role, and team (`PLAYER_IDENTIFIED`) live, per-frame, via the
incremental assigner in §1.7 — this was reprioritized to be resolved as early as
possible, ahead of the HTTP/WS layer in §3–5, precisely because so much else
(refinement, evaluation, possession/turnover, and most of `feature_service`
downstream — see §1.8) depends on team being correct.

---

## 1.7 Live team assignment — incremental, per-frame, adaptive

Supersedes the "deferred" decision above. Reuses `TeamClassifier._tracklet_features`
/ `_combined_feature_matrix` (the static helpers factored out of `team.py` during
the goalkeeper-assignment fix) for the underlying feature construction — same color
histogram + ReID embedding representation as the batch algorithm, applied
differently.

### 1.7.1 What the batch algorithm actually clusters (background, for the design below)

`team.py`'s existing KMeans input is a concatenation of a 32-d HSV color histogram
(`cv2.normalize`'d, which defaults to L2 normalization — i.e. unit-norm) and a
512-d ReID embedding (also L2-normalized, then scaled by 0.3). The `* 0.3` comment
calls this "30% weight," but Euclidean/KMeans distance is driven by *squared*
magnitude: the concatenated vector's squared norm is ~1² + 0.3² = 1.09, so color
contributes ~92% of the distance signal and ReID contributes ~8%, not 30% — the
comment overstates ReID's actual influence. This isn't necessarily wrong in
practice (kit color is a strong, appropriate signal; OSNet/Market1501 embeddings
were trained for pedestrian re-identification, not kit-color discrimination, so
their being lightly weighted may be close to correct anyway) — but the weighting
should be an explicit, documented, tunable config value rather than a misleading
magic number. The redesign below keeps both feature types and makes the weight a
named config key (`team_live.reid_weight`).

**Known risk carried forward, not solved here:** raw HSV histograms have no
lighting/white-balance correction. A shadow, floodlight change, or broadcast
camera auto-exposure shift can shift a real player's histogram. This matters more
once centroids adapt continuously (§1.7.3) than it did for the batch one-shot fit,
since a systematic drift could compound. Flagged as a risk to monitor once real
footage is available, not resolved in this document.

### 1.7.2 `PitchViewGate` — a reusable primitive, not just a bootstrap detail

A frame counts as "on the pitch" only if:
1. Pitch calibration succeeded for that frame (`calibrate.py` returned a usable
   homography), **and**
2. At least a configured minimum number of player/goalkeeper detections are
   present in that frame.

Both signals are already computed by the pipeline — no new model needed. This gate
is used in two places, not one:
- **Bootstrap accumulation** (§1.7.3) only counts gated frames toward the 1-second
  target — a pre-match crowd shot, graphics overlay, or replay does not corrupt or
  delay-for-no-reason the bootstrap; it's simply skipped, and the 1-second target
  is measured in gated-frame time, not wall-clock stream time. If the broadcast
  opens on a long pre-match sequence, bootstrap keeps extending until it collects
  1 real second of pitch-view frames, however long that takes.
- **Ongoing live operation**: track-birth classification and centroid updates
  (§1.7.4) are paused while the gate is closed (mid-match cutaway to a replay,
  the bench, the crowd, a graphics overlay) and resume once it reopens — so a
  cutaway doesn't spawn spurious "new tracks" that corrupt a centroid.

### 1.7.3 Bootstrap

- Collect appearance features (color histogram + ReID embedding, per §1.7.1) from
  every player tracklet observed during gated frames, until 1 second's worth
  (config: `team_live.bootstrap_window_s`, default `1.0`) of gated pitch-view time
  has been collected.
- Once collected, run a single KMeans(k=2) fit over the accumulated per-tracklet
  mean features (reusing `_tracklet_features`/`_combined_feature_matrix`) to get
  two initial centroids, labeled left/right by average pitch-x — identical
  labeling logic to the batch algorithm's step 3.
- These two centroids seed the adaptive running means in §1.7.4; they are **not**
  frozen afterward.
- **Minimum-diversity guard**: if the gated bootstrap window closes without enough
  distinct tracklets to plausibly represent two teams (config:
  `team_live.bootstrap_min_tracklets`, default `6`), the window is extended rather
  than forcing a fit on insufficient data — consistent with the non-pitch-camera
  handling above; both cases resolve the same way ("keep waiting for real
  pitch-view signal"), just gated by a different sufficiency check.
- Tracks born before bootstrap completes are held at `team = null`, state
  `"classifying"`, and evaluated retroactively against the centroids once
  bootstrap finishes, using whichever single frame of that track is used per
  §1.7.4 (no special-casing beyond "wait for centroids to exist").

### 1.7.4 Per-track classification — no accumulation window

- The first gated frame in which a `track_id` appears (role `player` or
  `goalkeeper`) is classified **immediately**, using that single frame's feature
  vector — no multi-frame buffering or averaging at track birth. This is the core
  redesign goal: per-frame wherever possible, not per-sequence.
- Confidence = symmetric relative margin between the two centroid distances:
  `confidence = (d_other - d_nearest) / (d_other + d_nearest)`. Ranges 0
  (equidistant, totally ambiguous) to close to 1 (unambiguous). **Note:** an
  earlier draft of this formula (`1 - d_nearest/(d_nearest+d_other)`) is
  algebraically equivalent to `d_other/(d_nearest+d_other)`, which — since
  `d_nearest` is by definition ≤ `d_other` — can never fall below 0.5,
  silently making any threshold below 0.5 unreachable and defeating the
  unconfirmed/retry path entirely. Caught by synthetic testing during
  implementation; the symmetric form above is the one actually implemented in
  `gsr_pipeline/team_live.py`.
- **Confidence ≥ threshold (config: `team_live.confidence_threshold`, default
  `0.25`, provisional — needs empirical tuning against real footage, same caveat
  as every other numeric default in this document) → locked permanently.** No
  future frame ever reclassifies that track, per the explicit requirement that
  classification happens once and stays fixed.
- **Confidence < threshold → left unconfirmed.** Retried every
  `team_live.reclassify_interval_s` (default `3.0`) seconds, using that track's
  *latest available single frame* at retry time (still no averaging — a fresh
  single sample each retry) — until it locks or the track ends without ever
  confirming.

### 1.7.5 Adaptive centroids — confidence-weighted running mean, no freezing

Every classification attempt (a track's birth-frame attempt, or an unconfirmed
track's periodic retry — **not** every frame of an already-locked track, see
below) updates the *assigned* team's centroid as a confidence-weighted running
mean:

```
W_team += confidence
S_team += confidence * feature_vector
centroid_team = S_team / W_team
```

Recommended over a fixed learning-rate update (`centroid += lr * confidence *
(x - centroid)`) because it needs no separate, hand-picked decay constant — a
low-confidence contribution barely moves the centroid, a high-confidence one moves
it more, and the estimate naturally stabilizes as accumulated weight grows. This
is a recommendation, not something explicitly specified — flag if a fixed-lr
scheme is preferred instead.

**Update cadence is tied to track lifecycle events, not raw frame rate**: only a
track's own classification-decision moments (birth, unconfirmed retries) push an
update. An already-locked, long-lived track does not keep re-pinging its centroid
every subsequent frame — otherwise a single long-tracked player would dominate
"the team's" centroid disproportionately relative to shorter-lived tracks (e.g. a
substitute), which isn't the intended meaning of a team appearance centroid. This
still keeps every individual contribution a single frame's raw feature vector
(never a multi-frame average) — the "no accumulation" goal is satisfied at the
feature-extraction level; only the *update cadence* is bounded, deliberately.

### 1.7.6 Concurrent ReID — background workers, not inline

`reid.py`'s embedding extraction currently runs synchronously inline in the
per-frame loop. In `process_stream`, this decouples:

- The main per-frame loop does detect → track only (fast) and can emit
  `PLAYER_POSITION`/`BALL_POSITION` immediately, without waiting on ReID.
- Detected crops are handed to a background worker pool (bounded queue + one or
  more worker threads) that computes embeddings and color features concurrently,
  publishing results back to the team-assignment/possession logic once ready.
- This directly extends the latency-decoupling principle `PIPELINE_EVOLUTION.md`
  §2 already established for event detection generally (don't gate fast signals
  behind slower stages) — ReID/team assignment becomes one more stage that fans
  out off tracking instead of blocking it.

### 1.7.7 Consequence for `TURNOVER` (possession.py) — now resolved, not a gap

Since team is live and per-frame from track birth (not deferred), `possession.py`'s
`TURNOVER` derivation (which requires a non-null `team`) works in live mode the
same way it already works in batch mode — no special-casing needed.  The one
residual edge case: a track that never confirms (stays `"classifying"` its whole
life, e.g. it's too short-lived to ever hit the confidence threshold) still
contributes `team = null` `POSSESSION` records while unconfirmed, same as the
pre-existing "clustering never ran" edge case already documented in
`possession.py`'s docstring — not a new gap, just the same one restated for the
live case.

### 1.7.8 Live-mode accuracy metric (build item, not designed in full here)

GS-HOTA (`evaluate.py`) only validates the batch pipeline's output against a
complete `predictions.json`. There is currently no equivalent for validating the
incremental live assigner. Proposed direction: a harness that replays a labeled
clip (e.g. the former `valid_sample` clip) through the *live* per-frame
path (`process_stream` + `LiveTeamAssigner`, driven by `VideoFileFrameSource` as
a stand-in feed) and compares its per-frame team assignments against
`Labels-GameState.json`, reusing `evaluate.py`'s existing ground-truth loading
code. Scoped as a real, separate build item (§8) — not designed to implementation
level in this document.

---

## 1.8 Everywhere team assignment is a load-bearing dependency

Compiled so the blast radius of any change to team-assignment logic is visible in
one place, not rediscovered piecemeal:

**Inside `cv_service`:**
- **`gsr_pipeline/refine.py`** — tracklet split/merge requires team equality as a
  hard gate (`t1["team"] != t2["team"]` blocks a merge; a team change mid-tracklet
  is a split trigger). Built for the whole-sequence batch case; not yet
  re-examined for how (or whether) refinement applies in streaming mode at all.
- **`gsr_pipeline/evaluate.py`** — GS-HOTA's `IdSim` requires an exact team match
  (with role and jersey) for a prediction to count as correctly identified. A
  wrong team assignment directly lowers the reported GS-HOTA score, not just a
  display label.
- **`gsr_pipeline/output.py`** — `team` is a required field in every
  `predictions.json` detection record and in `format_for_minimap`'s output.
- **`gsr_pipeline/possession.py`** — `POSSESSION.team` and `TURNOVER.team` are
  read directly off the possessor's `team`. A wrong or null team can silently
  suppress a `TURNOVER` that should have fired, or mislabel which team has the
  ball.
- **`docs/architecture/REQUIREMENTS.md`** — FR-005 (`PLAYER_IDENTIFIED` carries team), FR-006
  (turnover payload), the players-only clustering business rule (amended for
  goalkeepers), §6 Data Requirements.
- **`docs/architecture/PIPELINE_EVOLUTION.md`** — `PLAYER_IDENTIFIED`'s team field; `TURNOVER`'s
  entire definition is "did the possessing *team* change."

**Inside `data_ingestion`:**
- **`scripts/build_golden_fixture_from_gsr.py`** — converts the `"left"`/`"right"`
  team string into a numeric `team_id`, which then derives `player_id`
  (`team_id * 100 + jersey_number`, or a track_id-based fallback). A wrong team
  changes a player's *derived ID* in every downstream golden fixture built from it,
  not just a label.

**Downstream, in `feature_service`** (name-only grep this session, not a full
read — flagging that distinction explicitly): `team`/`team_id` appears across
roughly 118 files — formation classification, press-and-intensity, possession
tracking, and set-piece recognition all appear to key off team identity as a
first-class dimension. This wasn't independently verified in depth, but the
breadth alone means **team-assignment correctness is likely load-bearing for most
of BallKnowledge's tactical-analytics output**, not a peripheral detail. Worth a
dedicated read of `feature_service` before this ships, separate from this
document.

---

## 2. Stream lifecycle state machine

Governs both the HTTP API (§3) and `process_stream`'s internal state, single active
stream (per `docs/architecture/REQUIREMENTS.md` — v1 assumption):

```
IDLE ──register──▶ VALIDATING ──ok──▶ STREAMING ──stop/disconnect──▶ STOPPING ──▶ IDLE
                        │
                        └──fail──▶ IDLE   (no state change persists; source discarded)
```

- **IDLE**: no active stream. `POST /streams` accepted.
- **VALIDATING**: a `POST /streams` request is running `FrameSource.validate()`
  (§1.4). Another `POST /streams` received during this window is rejected with
  `409 Conflict` (validation-in-progress counts as "a stream is active" for the
  single-stream-at-a-time rule — see §3.1 for the full rejection rule).
- **STREAMING**: validation succeeded, `process_stream` is running in a background
  task, events are flowing to the WS endpoint (§4).
- **STOPPING**: `POST /streams/{id}/stop` received, or the `LiveBroadcastFrameSource`
  disconnected unexpectedly. `stop_event` is set; `process_stream`'s loop exits after
  its current frame; `FrameSource.close()` is called; any open WS connections
  receive a close frame (§4.4).
- Back to **IDLE** once teardown completes.

**In-flight sequence-level work at shutdown:** per `docs/architecture/REQUIREMENTS.md` FR-003,
this was flagged as unresolved. Decision for this milestone: since Phase 2
(tracklet aggregation, team, refinement) does not run in streaming mode at all (§1.6),
there is no in-flight sequence-level state to flush or discard at shutdown — this
question is moot for as long as team assignment stays deferred, and needs revisiting
only when that milestone lands.

---

## 3. HTTP API (cv-service, FastAPI)

### 3.1 `POST /streams`

Request:
```json
{
  "source_type": "rtmp" | "rtsp" | "hls" | "file",
  "source_url": "string"
}
```
`source_type` selects which `FrameSource` implementation to construct
(`LiveBroadcastFrameSource` for `rtmp`/`rtsp`/`hls`, `VideoFileFrameSource` for
`file` — the `file` option exists for local testing/staging, per §5, not as a
promised production input type; `docs/architecture/REQUIREMENTS.md` §12 still marks non-live
input as "Unknown" for whether it's ever a supported *production* capability).

Responses:
| Status | Condition | Body |
|---|---|---|
| `201 Created` | validation passed, stream now STREAMING | `{"stream_id": "...", "state": "streaming"}` |
| `409 Conflict` | a stream is already ACTIVE (VALIDATING or STREAMING) | `{"error": "stream_already_active", "active_stream_id": "..."}` |
| `422 Unprocessable Entity` | request body malformed (bad `source_type`, missing `source_url`) | `{"error": "invalid_request", "detail": "..."}` |
| `424 Failed Dependency` | source unreachable (`FrameSource.validate()` → `unreachable`) | `{"error": "source_unreachable", "detail": "..."}` |
| `422 Unprocessable Entity` (reused, distinguished by `error`) | source undecodable (`undecodable`) | `{"error": "source_undecodable", "detail": "..."}` |
| `422 Unprocessable Entity` (reused, distinguished by `error`) | no usable video (`no_usable_video`) | `{"error": "source_no_usable_video", "detail": "..."}` |

**Decision recorded, distinct from `docs/architecture/REQUIREMENTS.md`'s "Unknown / open"
rejection-semantics question:** a registration attempt while a stream is active is
now decided as an outright `409` rejection, not a queue or a replace — chosen for
simplicity and because it matches the "single active stream" v1 constraint most
directly. `docs/architecture/REQUIREMENTS.md` §3.1/§7 should be updated to reflect this once
implemented (currently still says "Unknown").

**Decided (approved):** the `4xx` code choices above — `424` for unreachable, `422`
reused for two different undecodable/black-screen reasons distinguished by an
`error` field rather than three separate codes, `409` for a registration conflict —
are confirmed as final for this milestone.

### 3.2 `POST /streams/{stream_id}/stop`

| Status | Condition |
|---|---|
| `200 OK` | stream stopped, state now IDLE |
| `404 Not Found` | `stream_id` doesn't match the currently active stream (includes the "no active stream" case from `docs/architecture/REQUIREMENTS.md` FR-003 — decided as `404` rather than a silent no-op `200`, so callers can distinguish "you stopped something" from "there was nothing to stop") |

**Both decisions above (409 for registration conflict, 404 for stop-with-nothing-active)
are approved and final for this milestone** — `docs/architecture/REQUIREMENTS.md`'s corresponding
"Unknown" items should be updated to match once implemented.

`source_type: "file"` (§3.1) is confirmed as a **development/staging-only** input,
not a promised production capability — `docs/architecture/REQUIREMENTS.md` §12 should continue to
mark non-live input as out-of-scope-for-production rather than be updated to promise it.

### 3.3 Auth

**Decided (recommendation, approved):** a static API-key header (e.g.
`X-CV-Service-Key`), value sourced from an environment variable (never committed),
validated via a FastAPI dependency on both `/streams` endpoints and the WS
handshake in §4. Sufficient for an internal service-to-service call between
`cv_service` and `data_ingestion` — not a public-facing endpoint — and upgradeable
to something stronger (mTLS, OAuth) later without changing the request/response
shapes above. `docs/architecture/REQUIREMENTS.md` §9 should be updated to reflect this once
implemented.

---

## 4. WebSocket event feed — ✅ Implemented (build-order step 7)

### 4.1 Endpoint

`GET /streams/{stream_id}/events` (WebSocket upgrade), implemented in `api/main.py`'s
`stream_events`. **Single consumer, confirmed**: a second concurrent connection attempt
to the same `stream_id` is rejected (close code 4409) rather than fanned out — matches
the "one intended client (`CVPipelineSource`)" assumption this section always had, made
an explicit enforced rule instead of just an assumption.

### 4.2 Message schema — implemented shape differs from the original proposal below

**What's actually on the wire today** (`api/event_serializer.py`), one JSON object per
WS message, mirroring `GSRPipeline.process_stream`'s existing "fast"/"appearance"
internal event shape rather than the full `event_type`/`payload` taxonomy proposed
below:

```json
// "fast" event -- emitted every frame, position/role only
{
  "type": "fast",
  "frame_idx": 1234,
  "timestamp_s": 49.36,
  "is_pitch_view": true,
  "detections": [
    {"track_id": 7, "role": "player", "role_confidence": 0.91,
     "bbox_ltwh": [x, y, w, h], "pitch_x": 12.4, "pitch_y": -3.1}
  ]
}

// "appearance" event -- emitted once per frame's ReID/jersey/team job completes
// (arrives later than, and out of strict order with, "fast" events for other frames --
// see §1.7.6/reid_worker_pool.py, unchanged by this step)
{
  "type": "appearance",
  "frame_idx": 1231,
  "detections": [
    {"track_id": 7, "role": "player", "role_confidence": 0.91,
     "bbox_ltwh": [x, y, w, h], "pitch_x": 12.4, "pitch_y": -3.1,
     "jersey_number": "10", "team": "left", "team_confidence": 0.62, "team_locked": true}
  ]
}
```

**This is a deliberate scoped-down v1, not a silent abandonment of the original
design below.** The original proposal (`event_type` matching `data_ingestion`'s
`EventType` enum 1:1, `POSSESSION`/`TURNOVER`/`PLAYER_IDENTIFIED` as first-class wire
events) requires deciding per-event-type `payload` shapes that `docs/architecture/REQUIREMENTS.md`
§6 still flags as undecided, and requires wiring `possession.py`'s detection logic into
`process_stream` at all — it currently only runs in the batch path (`process_sequence`).
Neither blocker needed solving to prove the transport layer (auth, single-consumer,
backpressure, lifecycle) end-to-end, so this step ships the raw per-frame detection
feed cv-service already produces internally. **Translating this into the
`event_type`/`payload`/`EventType` envelope — including deciding where possession/
turnover computation runs for live streams — is deferred to step 9
(`CVPipelineSource`)**, the actual consumer that needs that specific shape; it is the
right place to make that call because it's where the shape gets consumed, not
speculated about.

**What cv-service does NOT assign:** `sequence_number` and `source_id` — those remain
`DataEvent` fields owned by `BaseDataSource._make_event` on the `data_ingestion` side
(§5.1). cv-service's wire messages only carry `frame_idx` for ordering within its own
stream, unchanged from the original plan.

### 4.3 Connection lifecycle — implemented

- Auth: same static API key as the REST endpoints (`auth.py`), passed as a
  `?api_key=` query param on the WS handshake (not a header — most WS clients,
  including browsers, cannot set custom headers during the handshake). Missing/wrong
  key closes with code 4401.
- Client connects only after a successful `POST /streams` (needs the `stream_id`
  first) — confirmed, unchanged from the original plan.
- Connecting to a `stream_id` that isn't the currently active stream closes with
  code 4404.
- **If the WS connection drops while STREAMING, the pipeline keeps running** —
  confirmed, unchanged. `EventBroadcaster.publish` (called from the background
  `process_stream` thread) is a no-op write into a bounded queue; nothing about it
  can stall or kill the pipeline thread.
- **Decided and implemented: drop, no replay.** `EventBroadcaster` has no history
  buffer — a client that connects mid-stream only sees events from its connect-time
  onward. A *reconnecting* client after a genuine disconnect gets nothing from the
  gap either (same "drop" policy, not just "no buffer for late joiners") — this is a
  slightly stronger form of the original "drop" decision, worth flagging if step 9
  finds gap-tolerance actually matters in practice.
- **New (not in the original proposal): backpressure.** Bounded per-connection queue
  (`ws.max_queue_size`, default 100). A consumer that falls behind causes the OLDEST
  queued event to be dropped (not the newest, and not blocking `process_stream`'s
  frame loop) — same policy already used by `reid_worker_pool.py`'s job queue, applied
  here for consistency rather than picked independently.

### 4.4 Server-initiated close — implemented, close codes decided

Standard custom range (4000–4999), no longer TBD:

| Code | Meaning |
|------|---------|
| 4401 | Missing or invalid API key on the WS handshake |
| 4404 | `stream_id` doesn't match the currently active stream |
| 4409 | Another consumer is already attached to this stream's feed |
| *(none)* | Normal close (code 1000, the WS default) once the stream stops or the source is exhausted — no custom code needed here since there's nothing to distinguish beyond "the feed is over," and the client already knows via `POST /streams/{id}/stop`'s own response or by observing no more events arrive |

One implementation detail worth recording since it cost a debugging round-trip:
**the rejection close code is only delivered if the handshake is `accept()`-ed
first.** Closing before `accept()` makes uvicorn/Starlette reject the handshake at
the HTTP level (a bare `403`, no code or reason delivered to the client) rather than
completing the WS upgrade and sending a proper close frame — every rejection path in
`stream_events` therefore accepts, then immediately closes with the intended code.

Verified end-to-end with a real running server and a real WS client
(`api/validate_ws.py`, using the `websockets` library): **9/9 checks passing** —
4404 on a nonexistent stream, 4401 on a wrong key, real "fast"/"appearance" events
received with the documented shape, 4409 on a second concurrent consumer, and a clean
close once the stream is stopped via the REST endpoint.

---

## 5. `CVPipelineSource` (in `data_ingestion`) — ✅ Implemented (build-order step 9)

Supersedes `PIPELINE_EVOLUTION.md` §1's "in-process library import" plan for this
component specifically — everything else in that document (event taxonomy, Kafka
topic mapping) is unaffected. Lives at
`data_ingestion/src/data_ingestion/cv_pipeline_source.py`, follows the existing
`BaseDataSource` contract exactly (same shape as `AnnotatedReplaySource`,
`StatsBombSource`) — no changes needed to `base_source.py`, `registry.py`, or
`kafka_publisher.py` to support it.

```python
class CVPipelineSource(BaseDataSource):
    def __init__(self, source_id, cv_service_base_url, cv_source_type, cv_source_url,
                 api_key, match_id=None, cfg=None): ...

    def _start_streaming(self) -> None:
        # POST {cv_service_base_url}/streams (X-CV-Service-Key header) with
        # {"source_type": cv_source_type, "source_url": cv_source_url}.
        # Non-201 raises -- surfaced to whatever called connect(), per
        # BaseDataSource's existing convention.
        # On 201: launches one background thread owning a synchronous WS
        # connection (websocket-client) to /streams/{stream_id}/events.

    def _stop_streaming(self) -> None:
        # Signals the background thread to stop, joins it (closing the WS),
        # then POSTs /streams/{stream_id}/stop.
```

### 5.1 Wire-shape translation — the deferred decision, now made

§4.2 deferred exactly two decisions to this component: how cv-service's raw
`"fast"`/`"appearance"` WS messages become `EventType`/`DataEvent`s, and where
live-mode possession/turnover fits. Both are resolved in
`cv_pipeline_source.py`:

- **`"fast"` → `PLAYER_POSITION` / `BALL_POSITION`, one DataEvent per detection
  row**, split by `role` (`player`/`goalkeeper` → `PLAYER_POSITION`, anything
  else → `BALL_POSITION`). Genuinely per-frame/append-only, matching
  `PIPELINE_EVOLUTION.md` §3.1's stated semantics for position events — every
  `"fast"` message arrives every frame regardless of ReID/team readiness.
- **`"appearance"` → `PLAYER_IDENTIFIED`, but only on change.** cv-service's
  `"appearance"` messages arrive far more often than a track's *identity*
  meaningfully changes (every completed background job, including retries
  whose `team_confidence` merely wobbles per `team_live.py`'s running-mean
  updates). `CVPipelineSource` keeps a per-`track_id` `(team, jersey_number)`
  cache and only emits when that pair actually changes — implementing
  `PIPELINE_EVOLUTION.md` §3.1's "on first confirmation or on change, not
  every frame" requirement, which cv-service's own wire format does not
  enforce on its own.
- **`POSSESSION`/`TURNOVER`: still not wired — confirmed as a real gap, not
  silently dropped.** cv-service's `possession.py` only runs in the batch path
  (`process_sequence`); `process_stream` never computes it, so there is
  nothing on the wire for `CVPipelineSource` to translate yet. Documented in
  this file's module docstring as the concrete next piece of work if live
  possession/turnover is needed — it would need `possession.py`'s logic (or
  an incremental equivalent) wired into `process_stream` first, on the
  cv-service side, before anything changes here.

### 5.2 Reconnect policy — implemented, matches the original proposal

If the WS connection drops unexpectedly (not via `_stop_streaming`):
- Retries with exponential backoff (default 1s base, doubling, capped at 30s),
  up to a configured max-attempts count (default 5) — same policy (and same
  defaults) as `LiveBroadcastFrameSource`'s reconnect logic on cv-service's
  side of this same kind of problem, picked for consistency across both ends
  of the same architectural pattern.
- Each retry re-establishes the WS connection to the **same** `stream_id`
  (does not re-register — confirmed, the stream is still STREAMING on
  cv-service's side per §4.3).
- On exhausting retries: emits `EventType.SOURCE_DISCONNECTED` with
  `{"reason": "reconnect_attempts_exhausted"}` and gives up — does NOT
  automatically call `POST /streams/{id}/stop` (confirmed, remains an
  explicit, separate operation per FR-003).
- A clean server-initiated WS close (cv-service's own stream ending normally)
  is distinguished from a drop and does NOT trigger a reconnect attempt — the
  source's connection loop simply returns.

Verified end-to-end against a REAL running cv-service server (not mocked) via
`data_ingestion/src/data_ingestion/validate_cv_pipeline_source.py`: **7/7
checks passing** — registration, real `PLAYER_POSITION`/`BALL_POSITION` events
received with correct `match_id`/`source_id`/payload shape, monotonically
increasing `sequence_number`, clean `disconnect()` (stops cv-service's stream
too, confirmed via `/health` reporting idle afterward). No `PLAYER_IDENTIFIED`
events arrived in this particular run (short clip, default bootstrap window) —
noted, not treated as a failure; the translation logic itself was exercised
directly, just not triggered by this specific run's footage.

---

## 6. Config additions (proposal)

Following the existing `configs/default.yaml` convention (no hardcoded tunables —
see `possession` section precedent):

```yaml
frame_source:
  validation:
    max_frames: 30
    timeout_s: 10.0
    black_intensity_threshold: 10.0   # 0-255 scale; provisional, needs real-footage tuning
    frozen_diff_threshold: 1.0        # 0-255 scale, mean abs frame-to-frame diff; provisional

streaming:
  # process_stream() tunables -- distinct from the batch `possession` section,
  # since live mode may need different (e.g. looser) settings once team
  # assignment is added
  max_active_streams: 1               # enforces the v1 single-stream constraint

pitch_view_gate:
  min_player_detections: 4            # minimum player+goalkeeper detections for a
                                       # frame to count as "on the pitch" -- used by
                                       # both bootstrap accumulation and ongoing
                                       # cutaway handling (§1.7.2)

team_live:
  # Incremental live team assignment (§1.7) -- distinct from the batch `team`
  # section, since the live algorithm is structurally different (per-frame,
  # adaptive centroids vs. one-shot whole-sequence KMeans).
  reid_weight: 0.3                    # named/tunable now instead of a magic number
                                       # in code -- see §1.7.1 for why the *effective*
                                       # influence is much lower than this value alone
                                       # suggests (squared-distance dynamics)
  bootstrap_window_s: 1.0             # gated pitch-view seconds needed to seed centroids
  bootstrap_min_tracklets: 6          # minimum distinct tracklets before attempting the fit
  confidence_threshold: 0.25          # relative-margin threshold to lock a track's team; provisional
  reclassify_interval_s: 3.0          # retry interval for unconfirmed tracks
```

---

## 7. Open questions — status

Most items previously listed here are now resolved by decisions recorded above.
What remains genuinely open:

1. ~~Auth~~ — **resolved** (§3.3: static API-key header).
2. ~~Exact `4xx`/close-code taxonomy~~ — **resolved** (§3.1, §3.2: confirmed final).
3. ~~WS disconnect buffering~~ — **resolved** (§4.3: drop).
4. **Validation thresholds** (§1.4, §6) — defaults set, but genuinely provisional:
   need tuning against real broadcast footage, which doesn't exist in this repo yet.
   Not something to "resolve" by decision — needs actual sample footage.
5. ~~Live-mode team/side assignment~~ — **resolved**, fully designed (§1.7).
   Numeric defaults within it (`confidence_threshold`, etc.) carry the same
   footage-dependent caveat as #4.
6. ~~WS exact close-code values~~ — **resolved** (§4.4: 4401/4404/4409, implemented
   and verified).
7. ~~`docs/architecture/REQUIREMENTS.md` needs updating once this lands~~ — **resolved**:
   registration-conflict, shutdown-with-no-active-stream, and auth are all
   now reflected there with Implemented status lines (§14 traceability, §5
   API table, §9 Security).
8. **Live-mode accuracy metric** (§1.7.8) — direction proposed, not designed to
   implementation level; a real scoped follow-up item, not a blocker for starting
   §8's build order.
9. **`feature_service`'s actual dependence on `team`/`team_id`** (§1.8) — breadth
   established by grep, depth not verified this session. Worth a dedicated read
   before the live team assigner ships, to catch any downstream assumption (e.g.
   about team assignment being final/non-retroactive) this design might violate.

---

## 8. Implementation build order (updated: team assignment prioritized earliest)

1. ✅ **Done.** `FrameSource` ABC + `ImageDirFrameSource` + `pipeline.py` refactor
   (§1.1, §1.2, §1.5). Regression-verified: pre-/post-refactor `predictions.json`
   on the then-committed `valid_sample` sequence are byte-for-byte identical.
2. ✅ **Done.** `VideoFileFrameSource` + `validate()` default implementation
   (§1.4), with default thresholds from §6. Verified against a real annotated
   `.mp4` (validates OK) and a missing file (`unreachable`), plus synthetic
   black/frozen/normal frame batches for the detection logic itself.
3. ✅ **Done.** `PitchViewGate` (§1.7.2) + `LiveTeamAssigner` (§1.7.3–1.7.5),
   unit-tested against synthetic tracklet data: gated bootstrap correctly
   ignores non-pitch frames and extends when insufficient tracklet diversity
   is seen, clearly-separated tracks lock immediately, ambiguous tracks stay
   unconfirmed and retry, centroids measurably adapt. **A real bug was found
   and fixed during this testing**: the original confidence formula
   (`1 - d_near/(d_near+d_other)`) could never fall below 0.5 by construction,
   which would have made the documented default threshold (0.25) unreachable
   and silently defeated the entire unconfirmed/retry path. Corrected to the
   symmetric margin `(d_other-d_near)/(d_other+d_near)` — see §1.7.4's updated
   note and `team_live.py`'s inline comment.
4. ✅ **Done.** `process_stream` (§1.6) + concurrent `ReidWorkerPool` (§1.7.6)
   built together (built concurrently from the start, not as a later
   performance follow-up — revised from the original plan), wired to
   `PitchViewGate` + `LiveTeamAssigner` from step 3. `_detect_track_calibrate`
   extracted as a method shared with `process_sequence` (single source of
   truth for the fast-path stages) — regression-verified byte-identical
   batch output before/after via the same stash/run/diff methodology used
   for step 1. `process_stream` itself verified end-to-end against the REAL
   model stack (not synthetic data) driven by a real `.mp4` via
   `VideoFileFrameSource` as the live-source stand-in — see
   `gsr_pipeline/validate_process_stream.py`, 7/7 checks passing.
5. ✅ **Done.** Live-mode accuracy harness (§1.7.8,
   `gsr_pipeline/evaluate_team_live.py`), run against 20 real frames of
   `valid_sample` through the step-4 live path (`process_stream` +
   `LiveTeamAssigner`), matched against `Labels-GameState.json`. First real
   numbers (small sample, `bootstrap_window_s` overridden to 0.2s purely to
   keep this harness's own runtime bounded on CPU — not a claim about the
   production default): **292 GT-matched player/goalkeeper detections, 234
   locked (80.1% lock coverage), 100.0% accuracy among locked predictions,
   80.1% among all matched** (unconfirmed counted as wrong by definition,
   since their `team` is `null`). Treat as an encouraging first signal, not a
   validated production accuracy figure — 20 frames from one sequence is not
   a representative benchmark.
6. ✅ **Done.** FastAPI app (`api/main.py`, `api/stream_manager.py`,
   `api/schemas.py`, `api/auth.py`): `POST /streams`, `POST /streams/{id}/stop`,
   `GET /health`, backed by step 4's `process_stream`. Implements the full
   state machine from §2 (IDLE/VALIDATING/STREAMING/STOPPING, single active
   stream), the status-code table from §3.1-3.2 exactly, and the static
   API-key auth from §3.3. `source_type` at the time this step shipped routed
   ALL of `rtmp`/`rtsp`/`hls`/`file` through `VideoFileFrameSource` since
   `LiveBroadcastFrameSource` didn't exist yet — **superseded by step 8**,
   which gave `rtmp`/`rtsp`/`hls` their own real reconnect-on-drop semantics;
   only `file` still uses `VideoFileFrameSource` today. Verified end-to-end
   with a real running server (`uvicorn`) and real HTTP calls via
   `api/validate_api.py`: **11/11 checks passing**, covering auth
   (missing/wrong key), unreachable-source rejection (424), successful
   registration (201), registration-conflict rejection (409), stop-nonexistent
   (404), successful stop (200), and a clean re-registration after stop.
7. ✅ **Done.** WS event feed (§4, `api/event_broadcaster.py`,
   `api/event_serializer.py`, `stream_events` in `api/main.py`) wired to the same
   running stream via `StreamManager`'s background thread. Single consumer, drop-
   oldest backpressure, same static API-key auth as REST (query param), close codes
   4401/4404/4409 decided and implemented. Ships the pipeline's existing raw
   "fast"/"appearance" per-frame shape rather than the originally-proposed
   `event_type`/`EventType`/`payload` taxonomy — see §4.2 for why that's a
   deliberate, documented scope decision deferred to step 9, not a dropped
   requirement. Verified end-to-end with a real server + real WS client
   (`api/validate_ws.py`): **9/9 checks passing**.
8. ✅ **Done.** `LiveBroadcastFrameSource` (§1.2, `gsr_pipeline/frame_source.py`) —
   real RTMP/RTSP/HLS via OpenCV's FFmpeg backend, with transparent
   reconnect-on-drop inside `read_frame()` (exponential backoff, capped
   attempts) rather than surfacing every transient failed read as
   end-of-stream. `disconnect_reason="reconnect_attempts_exhausted"` once
   retries are exhausted, distinguishing a genuine drop from
   `VideoFileFrameSource`'s "sequence complete" `None`. `frame_idx` keeps
   counting across a reconnect (not reset); `timestamp_ms` is
   wall-clock-since-connect, not `CAP_PROP_POS_MSEC` (unreliable for live
   sources via OpenCV). Wired into `api/stream_manager.py`'s
   `_build_frame_source`: `file` → `VideoFileFrameSource` (unchanged),
   `rtmp`/`rtsp`/`hls` → `LiveBroadcastFrameSource` (no longer routed through
   the file stand-in). New `live_broadcast` config section
   (`reconnect_max_attempts`, `reconnect_backoff_base_s`,
   `reconnect_backoff_max_s`, `read_timeout_s`). **Verified via synthetic
   failure injection only** (`gsr_pipeline/validate_live_broadcast.py`,
   monkeypatching `cv2.VideoCapture` with a scripted fake — 12/12 checks
   passing: happy path, transient-drop-then-recover, exhausted-retries
   disconnect, recovery on a later attempt, connect-failure surfacing) — no
   real RTMP/RTSP/HLS feed exists in this dev environment to validate
   against, and per this session's standing constraint, real
   network-dependent evaluation belongs on the GPU machine, not here. This
   proves the reconnect/backoff/disconnect_reason state machine is correct;
   it does NOT prove OpenCV's FFmpeg backend behaves as assumed against a
   real live source — flagged as a genuine gap, not silently assumed away.
9. ✅ **Done.** `CVPipelineSource` in `data_ingestion` (§5,
   `data_ingestion/src/data_ingestion/cv_pipeline_source.py`), consuming the WS
   feed from step 7. Resolved both decisions deferred from §4.2: the raw
   "fast"/"appearance" wire shape is translated into
   `PLAYER_POSITION`/`BALL_POSITION`/`PLAYER_IDENTIFIED` DataEvents (the last
   deduped to fire only on actual identity change, per
   `PIPELINE_EVOLUTION.md` §3.1); `POSSESSION`/`TURNOVER` confirmed still not
   wired (cv-service doesn't compute them in live mode yet — a real gap, not
   a translation gap). Reconnect policy matches the original §5.1 proposal
   (exponential backoff, same `stream_id`, `SOURCE_DISCONNECTED` on
   exhaustion, no automatic stop). Verified end-to-end against a real running
   cv-service server:
   `data_ingestion/src/data_ingestion/validate_cv_pipeline_source.py`, 7/7
   checks passing.
   **Fixed 2026-09-13:** the translation treated every non-player/goalkeeper role
   as the ball, so all referee detections were published as `BALL_POSITION`.
   Every role is now mapped explicitly (`REFEREE_POSITION` added, unknown roles
   skipped with a warning), covered by `data_ingestion/tests/test_cv_pipeline_source.py`
   (4/4, confirmed to fail against the old mapping). The validator gained an
   eighth check ("no referee detection is published as BALL_POSITION"); it has
   not yet been re-run against a live cv-service server.
10. Update `docs/architecture/REQUIREMENTS.md`'s now-resolved "Unknown" items (§7.7 above).

Steps 1–9 are all done. Only step 10 remains: sweeping `docs/architecture/REQUIREMENTS.md` for
now-resolved "Unknown" items (§7.7). The one genuine functional gap left in the
whole pipeline is live-mode `POSSESSION`/`TURNOVER` — `possession.py` still only
runs in the batch path; wiring an incremental equivalent into `process_stream`
(cv-service side) is the natural next scoped feature, not a build-order step.
