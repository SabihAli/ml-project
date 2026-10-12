# Requirements — cv-service

Evidence key used throughout this document:

- **Confirmed** — explicitly documented in this repo (README, AGENTS.md, PIPELINE_EVOLUTION.md) or established by code/tests.
- **Observed** — currently implemented in code but not documented as an intended product requirement.
- **Inferred** — reasonably derived from stated goals but not explicitly established anywhere.
- **Stated (product owner)** — provided directly by the repository owner during this documentation task (2026-08-04), not yet reflected in code or prior docs.
- **Unknown** — cannot be determined from the repository as it stands.

`cv-service` (repo directory `cv_service/`, package `gsr_pipeline`, upstream remote `github.com/SabihAli/ml-project`) originally shipped as an **offline batch CLI pipeline only**, with no API layer, no streaming ingestion, and no Kafka integration (Confirmed at the time this document was first written — `cv_service/AGENTS.md`). That has since evolved substantially: a live-mode streaming path (`gsr_pipeline`'s `process_stream`, `FrameSource`, `LiveTeamAssigner`, `PitchViewGate`), an `api/` HTTP+WS layer implementing FR-001/002/003 plus a WS event feed, and a `data_ingestion`-side consumer (`CVPipelineSource`) that translates that feed into Kafka-bound `DataEvent`s now all exist and are verified — see those requirements' updated Status lines and `docs/architecture/STREAMING_ARCHITECTURE.md` §3-5. The one genuine functional gap remaining in this whole chain: live-mode `POSSESSION`/`TURNOVER` are not computed (`possession.py` only runs in cv-service's batch path) — confirmed, not a translation issue. Everywhere else in this document, the Confirmed/Observed/Stated/Inferred/Unknown distinction is preserved as originally written — do not assume a requirement is implemented just because this paragraph mentions the pipeline now exists end-to-end; check each requirement's own Status line.

---

## 1. Product/System Overview

**cv-service** is the computer-vision microservice of the BallKnowledge platform. BallKnowledge is a web application for football coaches, managers, and assistants that provides real-time analytics and strategic/tactical insight during a match (Stated — product owner).

- **Problem it solves**: coaching staff need live tactical insight (positions, roles, possession, on-ball actions) during a match, derived purely from the broadcast video feed — no dedicated tracking hardware, no manual tagging.
- **Input**: a live football broadcast stream, and only that (Stated — product owner). No other sensor/data feed is an input to this service.
- **Primary users/consumers**: the BallKnowledge frontend (on behalf of coaches/managers/assistants), and downstream BallKnowledge services (`data_ingestion`, and transitively `feature_service`) that consume the structured events cv-service produces (Confirmed intent — `PIPELINE_EVOLUTION.md`).
- **Major capabilities (target)**:
  - Accept and validate a broadcast stream source, register it for processing.
  - Run a CV pipeline that performs detection, tracking, role/team classification, pitch calibration, and (later) action/event classification, continuously, per live frame and per rolling sequence.
  - Emit structured events describing entities and match actions to the rest of the BallKnowledge system.
  - Allow the stream/pipeline to be stopped on demand.
- **Major capabilities (currently implemented, batch-only)**: end-to-end detection, tracking, pitch calibration, team/role classification, jersey OCR, tracklet refinement, and GS-HOTA evaluation, run once over a complete, pre-recorded sequence of frame images and written to a `predictions.json` file (Confirmed — `README.md`, `AGENTS.md`).

---

## 2. Actors / Users

| Actor | Role | Permissions |
|---|---|---|
| **BallKnowledge frontend** | The sole client of cv-service's exposed API. Registers stream sources and requests shutdown on behalf of an authenticated coaching-staff session. | Register a stream source; stop an active stream. Authenticates via a static API-key header (`X-CV-Service-Key`) — implemented in `api/auth.py`, decided in `docs/architecture/STREAMING_ARCHITECTURE.md` §3.3 as sufficient for an internal service-to-service call; upgradeable later. |
| **Coach / manager / assistant (end user)** | Human user of the BallKnowledge web app; does not call cv-service directly. | Indirect only, via the frontend. |
| **`data_ingestion` service** | Downstream consumer of cv-service's structured events; owns Kafka publishing for the whole system (Confirmed — `data_ingestion/README.md`, `PIPELINE_EVOLUTION.md`). | Receives events via a network *push* — `data_ingestion`'s `CVPipelineSource` is an HTTP+WS client of cv-service's `api/` layer, not an in-process library import (Implemented, resolved — see §5). |
| **`feature_service` / other downstream consumers** | Consume structured event data further downstream of `data_ingestion`. | Never access cv-service or raw video directly (Confirmed constraint — `PIPELINE_EVOLUTION.md` §1: "everything downstream... consumes structured JSON/event data only, never pixels"). |
| **Operator / SRE** | Deploys and operates the service; consumes observability signals. | Unknown — no deployment tooling (Docker, CI, IaC) exists in this repo yet (Confirmed absence — `AGENTS.md`). |

---

## 3. Functional Requirements

### 3.1 Stream lifecycle

#### FR-001 — Stream source registration
**Status:** Implemented (`api/main.py`, `api/stream_manager.py`) per the design decided in `docs/architecture/STREAMING_ARCHITECTURE.md` §3.1.

**Description:** The frontend registers a live broadcast stream source with cv-service by calling an exposed endpoint.

**Expected behavior:** On receiving a registration request, the system shall validate the stream source before accepting it. Validation is not a formality — it shall actually attempt to connect to and read from the source. Only after validation succeeds shall the stream be accepted for processing.

**Inputs:** `POST /streams` — `{"source_type": "rtmp"|"rtsp"|"hls"|"file", "source_url": "..."}`. `"file"` is dev/staging-only, not a promised production input (see §12).

**Outputs:**
- Success: HTTP `201` with `{"stream_id": "...", "state": "streaming"}`.
- Failure: `422` (malformed request or undecodable/no-usable-video source), `424` (unreachable source), or `409` (a stream is already active) — see FR-002.

**Constraints:** Only one stream may be active at a time in v1 — enforced in `api/stream_manager.py`. A registration request received while a stream is already active is rejected outright with `409 {"error": "stream_already_active", "active_stream_id": "..."}` (decided in `docs/architecture/STREAMING_ARCHITECTURE.md` §3.1, superseding the earlier "Unknown / open" note).

**Acceptance criteria:**
- [x] Registering a reachable, decodable, non-blank stream source returns `201` and the pipeline begins processing. — verified via `api/validate_api.py` against a real running server.
- [x] Registering while a stream is already active is rejected with `409`. — verified.
- [x] No pipeline processing begins for a source that fails validation. — verified (validation runs synchronously before the background thread starts).

---

#### FR-002 — Stream source validation
**Status:** Implemented (`gsr_pipeline/frame_source.py`'s `FrameSource.validate()`, wired into `api/stream_manager.py`) per `docs/architecture/STREAMING_ARCHITECTURE.md` §1.4, §3.1.

**Description:** Before accepting a registration, the system performs a battery of checks against the proposed stream source and reports a specific, distinguishable failure reason if any check fails.

**Expected behavior:** Validation includes, in order: `connect()` success (else `unreachable`), decoding at least one frame within `max_frames`/`timeout_s` (else `undecodable`), and a black/frozen-frame check across the sampled frames (else `no_usable_video`). Thresholds live in `configs/default.yaml`'s `frame_source.validation` section — **explicitly provisional**, not yet tuned against real broadcast footage (none exists in this repo).

**Inputs:** Same stream source reference as FR-001.

**Outputs:** `201` on full success; otherwise `424 {"error": "source_unreachable", ...}`, or `422` with `{"error": "source_undecodable", ...}` / `{"error": "source_no_usable_video", ...}` (same status code, distinguished by the `error` field — decided in `docs/architecture/STREAMING_ARCHITECTURE.md` §3.1, approved).

**Constraints:** Validation completes and responds before the pipeline is considered active — confirmed: `api/stream_manager.py`'s `register()` calls `frame_source.validate()` synchronously, in the request-handling thread, before starting the background `process_stream` thread.

**Acceptance criteria:**
- [x] An unreachable stream source is rejected with `424`, not a timeout/hang. — verified via `api/validate_api.py`.
- [ ] A reachable but non-decodable source is rejected with `422`/`source_undecodable`. — implemented, not yet exercised by a real non-decodable file in validation.
- [ ] A reachable, decodable source producing a black/frozen/unusable feed is rejected with `422`/`source_no_usable_video`. — implemented (unit-validated against synthetic black/frozen frame batches in `gsr_pipeline/validate_frame_source.py`), not yet exercised via the live HTTP endpoint with a real black/frozen video file.
- [x] A valid, usable stream source is accepted. — verified via `api/validate_api.py` against a real `.mp4`.

---

#### FR-003 — Stream shutdown
**Status:** Implemented (`api/main.py`, `api/stream_manager.py`) per `docs/architecture/STREAMING_ARCHITECTURE.md` §3.2.

**Description:** The frontend can stop the currently active stream and halt the CV pipeline via a second exposed endpoint.

**Expected behavior:** On request, the system shall stop reading from the stream source, halt all pipeline processing for it, release associated resources, and stop emitting new events for that stream.

**Inputs:** `POST /streams/{stream_id}/stop` — the `stream_id` returned by FR-001's registration.

**Outputs:** `200 {"stream_id": "...", "state": "idle"}` on success; `404 {"error": "stream_not_found", ...}` if `stream_id` doesn't match the currently active stream, **including the "no active stream" case** (decided in `docs/architecture/STREAMING_ARCHITECTURE.md` §3.2 — a `404`, not a silent no-op `200`, so callers can distinguish "you stopped something" from "there was nothing to stop").

**Constraints:** Shutdown behavior for in-flight sequence-level computation is resolved as moot for this milestone: `process_stream` runs Phase 1 only (no tracklet aggregation/batch team/refinement), so there is no in-flight sequence-level state to flush or discard at shutdown (see `docs/architecture/STREAMING_ARCHITECTURE.md` §2). `stop()` sets a `threading.Event` and joins the background thread (30s timeout) before returning.

**Acceptance criteria:**
- [x] Calling shutdown while a stream is active stops pipeline processing and frees resources. — verified via `api/validate_api.py`.
- [x] Calling shutdown when no stream is active returns `404`, not a crash. — verified.
- [x] After shutdown, a new registration (FR-001) can succeed. — verified (state machine fully clears back to IDLE).

---

### 3.2 Detection and analytics

#### FR-004 — Per-frame entity detection and localization
**Status:** Confirmed (implemented, batch mode) / Inferred (real-time target).

**Description:** For each processed video frame, the system detects and localizes on-pitch entities.

**Expected behavior:** For every frame, the system shall detect players, goalkeepers, referees, and the ball; assign each a bounding box; and (once pitch calibration succeeds) project each entity's position into real-world pitch coordinates.

**Inputs:** One video frame.

**Outputs:** Per-frame detections, each with entity role (player/goalkeeper/referee/ball), pixel bounding box, and pitch-relative coordinates (meters) where calibration is available.

**Constraints:** Role classification is owned entirely by the detector's 4-class output — no downstream heuristic re-derives player/goalkeeper/referee role (Confirmed — `AGENTS.md` Architecture Rule 7).

**Acceptance criteria:**
- [ ] Every frame with at least one visible entity produces at least one detection record.
- [ ] Each detection carries a role, a bounding box, and (when calibration succeeds) pitch coordinates.

---

#### FR-005 — Entity identity: tracking, team, jersey number
**Status:** Confirmed (implemented, batch mode, whole-sequence only) / Inferred (real-time target, per PIPELINE_EVOLUTION.md).

**Description:** The system maintains a consistent identity for each tracked entity across frames, and determines each player's team side and jersey number.

**Expected behavior:** Detections shall be associated across frames into tracks. For each player track, the system shall determine team side (via appearance clustering) and jersey number (via OCR), converging on one consistent value per track rather than flickering per frame.

**Inputs:** Sequence of per-frame detections for a given entity.

**Outputs:** A `PLAYER_IDENTIFIED`-style record (track ID, team, jersey number, role) emitted independently of position, "on first confirmation or on change, not every frame" (Confirmed target design — `PIPELINE_EVOLUTION.md` §3.1).

**Constraints:** In the current batch implementation, team classification and jersey/role consensus require the **entire** sequence/tracklet before producing a final answer (Confirmed — `AGENTS.md` "Batch-only design vs. real-time goal"). Making this incremental for live streams is explicitly flagged in `PIPELINE_EVOLUTION.md` as requiring genuine redesign, not a config change — no committed design exists yet for how "first confirmation" is determined in a live/incremental setting.

**Acceptance criteria:**
- [ ] Each player entity is eventually assigned a team and (when OCR succeeds) a jersey number.
- [ ] Identity for a given track does not flicker within a single finalized window/tracklet.

---

#### FR-006 — Possession and turnover detection
**Status:** Stated in target design (Confirmed intent — `PIPELINE_EVOLUTION.md` §3.1); not implemented.

**Description:** The system continuously determines which team/track currently possesses the ball, and emits a discrete event when possession changes team.

**Expected behavior:** `POSSESSION` is a continuous, rule-based signal (e.g., nearest player to ball) computed and emitted as a first-class event, not just internal state. `TURNOVER` is derived by comparing the current and previous possessor's team.

**Inputs:** Per-frame track + ball position data.

**Outputs:** `POSSESSION` events (ongoing) and `TURNOVER` events (discrete, on team change).

**Constraints:** No separate `POSSESSION_CHANGE` event exists — this was explicitly considered and rejected as redundant with `TURNOVER` (Confirmed — `PIPELINE_EVOLUTION.md` §3.1 note).

**Acceptance criteria:**
- [ ] A `TURNOVER` is emitted if and only if ball possession moves from a track on one team to a track on the other team.
- [ ] `POSSESSION` reflects the current possessor without requiring a `TURNOVER` to have fired.

---

#### FR-007 — Classifier-based action events (pass, shot, carry, dribble, tackle, touch, interception, duel)
**Status:** Confirmed target design (`PIPELINE_EVOLUTION.md` §3.2), explicitly deferred milestone; not implemented.

**Description:** The system detects on-ball actions that require visual motion context beyond position/geometry alone, using a short video clip cropped around the ball possessor (or referee, for a later deferred milestone).

**Expected behavior:** Two window sizes run in parallel: a short window (~0.5–1s) for fast/discrete actions, and a longer window (~2–3s) for actions needing before/after disambiguation. The long-window result may arrive later as a refinement/confirmation of an earlier fast prediction (same pattern used for `INTERCEPTION` refining an earlier `TURNOVER`). Exact frame-count parameters are explicitly not yet decided — to be set empirically (Confirmed — `PIPELINE_EVOLUTION.md` §2).

**Inputs:** A cropped clip sequence around the relevant entity.

**Outputs:** One of `PASS`, `SHOT`, `CARRY`, `DRIBBLE`, `TACKLE`, `TOUCH`, `INTERCEPTION`, `DUEL`.

**Constraints:** `CARRY` (uncontested possession + movement) and `DRIBBLE` (possession + movement + beating a defender) are distinct and both in scope — this was explicitly evaluated and resolved, not an oversight (Confirmed — `PIPELINE_EVOLUTION.md` §3.3). `TOUCH` is deliberately **not** rule-based (a proximity heuristic was explicitly rejected for false positives/negatives) and requires the classifier.

**Acceptance criteria:**
- [ ] No acceptance criteria yet — model/architecture not built. Flagged as not implementable until the classifier milestone begins.

---

#### FR-008 — Referee-crop / stoppage events (fouls, cards, corners, goal kicks)
**Status:** Confirmed as explicitly out of v1 scope (`PIPELINE_EVOLUTION.md` §3.3), deferred second milestone.

**Description:** Same clip-classifier infrastructure as FR-007, applied to a referee crop instead of the ball possessor, for foul/card/corner/goal-kick detection.

**Expected behavior:** Deferred until after ball-event classification (FR-007) is working end-to-end.

**Acceptance criteria:** Not applicable — explicitly deferred.

---

#### FR-009 — Broadcast graphics OCR (score, match clock)
**Status:** Stated (product owner); not implemented; not present in `PIPELINE_EVOLUTION.md`.

**Description:** The pipeline shall detect and read broadcast on-screen graphics — at minimum the match score and match clock — via OCR, as a further evolution beyond `PIPELINE_EVOLUTION.md`'s current plan.

**Expected behavior:** Unknown/not designed — no detection method, event type, cadence, or accuracy expectation has been specified anywhere. This is recorded here as a stated future capability, not a designed requirement.

**Acceptance criteria:** Cannot be written yet — no design exists.

---

### 3.3 Output

#### FR-010 — Structured event output (no raw pixels downstream)
**Status:** Confirmed as an architectural constraint (`PIPELINE_EVOLUTION.md` §1).

**Description:** All CV-derived information leaving cv-service is structured, typed event data — never raw frames/pixels.

**Expected behavior:** Any event requiring visual context (e.g., action classification) must be fully resolved inside cv-service before being emitted; downstream services never receive video.

**Constraints:** This is treated as a hard boundary — "any event that requires visual context... must be computed here; it cannot be pushed to a downstream service that only sees structured data" (Confirmed — `PIPELINE_EVOLUTION.md` §1).

**Acceptance criteria:**
- [ ] No API or integration path exists (or is planned) for downstream services to receive raw video/frames from cv-service.

---

## 4. Non-Functional Requirements

### Performance
Not specified. No latency budget, frame-rate target, or throughput number exists anywhere in the repository. The only latency-relevant statement is qualitative: classifier-based event detection requires a sliding window of frames and that latency is "irreducible, only tunable via window size" (Confirmed — `PIPELINE_EVOLUTION.md` §2), and window sizes themselves are explicitly not yet decided.

### Scalability
Not specified for production. v1 explicitly assumes a single active stream (Stated — product owner). Multi-stream/multi-match concurrency is not designed.

### Reliability
Not specified. No retry, failover, or crash-recovery behavior is documented for a live stream disconnecting mid-match, an inference stage erroring mid-sequence, or the pipeline process crashing.

### Availability
Not specified. No uptime target, health-check contract, or SLA exists.

### Security
Not specified beyond a general architectural note that only cv-service touches raw video (§3.3, FR-010). No authentication/authorization scheme for the registration/shutdown endpoints is documented. See §9.

### Privacy
Not specified. Broadcast video of a public football match is the sole input; no handling of personally identifiable data (e.g., biometric-style face data) beyond jersey/role/position is discussed anywhere.

### Maintainability
Confirmed conventions exist at the code level (config-driven construction, DataFrame contract, stage-ordering rules — see `AGENTS.md` "Architecture Rules"), but these are implementation conventions, not product requirements, and are out of scope for this document per the task's instruction not to duplicate `ARCHITECTURE.md`-type content.

### Observability
Stated (product owner) as a top-priority requirement, not yet implemented:
- High observability is required given the system's real-time, high-throughput nature.
- Logging/tracing must **not** be spammy at per-frame granularity, since per-frame logs "are no longer meaningful" at this throughput.
- **Aggregation-based** and **anomaly-based** logging/signals are required in place of naive per-event logging.
- No concrete metrics, trace spans, log fields, or alert thresholds are defined yet — these must be designed (the product owner pointed to the `observability-setup` skill as the mechanism for producing that design; it is a separate deliverable from this requirements document).

### Resource Constraints
Not specified in terms of hardware targets. Observed only: the existing batch pipeline is built around "nano-scale" models (YOLOv8n, OSNet-x0.25) specifically to minimize compute vs. the heavier official SoccerNet-GSR baseline (Confirmed — `AGENTS.md`, `docs/architecture/GSR_PIPELINE.md`), which is suggestive of a low-compute-footprint design goal, but no explicit CPU/GPU/memory budget is stated for the live service.

### Synchronization strategy (architecture consideration, not a settled requirement)

The system produces both **per-frame** detections (position, identity) and **per-sequence** detections (classifier-based actions, resolved only after a multi-frame window completes). Reconciling their different latencies for a coherent downstream/user experience is an open design question. Options, as raised during this documentation task (Stated — product owner) — presented here as trade-offs for a future design decision, not as a chosen approach:

1. **Emit-as-ready, no reconciliation (fully async).**
   Per-frame events stream immediately; per-sequence/classifier events arrive later, tagged with the frame range they cover, and are appended once ready.
   - *Latency:* lowest possible — position data is never held back.
   - *UX effect:* the frontend/timeline must be able to retroactively annotate a moment already shown to the coach (e.g., "that possession sequence was, in fact, a `DRIBBLE`"). Requires UI capable of updating past timeline entries.
   - *Best fit:* live tactical view where instantaneous position/possession matters more than immediately knowing the exact action label.

2. **Two-tier fast/slow emission with explicit refinement (as sketched in `PIPELINE_EVOLUTION.md`).**
   A short window (~0.5–1s) emits a fast, provisional classification; a longer window (~2–3s) later emits a refined/confirming event referencing the same action (the pattern already chosen for `INTERCEPTION` refining `TURNOVER`).
   - *Latency:* fast tier close to per-frame latency; slow tier adds ~1–2s beyond that.
   - *UX effect:* consumers must handle "provisional then confirmed" pairs and decide whether to show a fast guess that might later be corrected, or wait.
   - *Best fit:* matches the two-window design already decided for classifier events (§3.2, FR-007) — likely the least new design work since it's already partially specified.

3. **Buffered/held synchronization (wait for sequence-level detections before releasing a batch of frame-level events).**
   Hold per-frame events for the duration of the longest relevant window, then release a synchronized batch covering both tiers together.
   - *Latency:* highest — every event is delayed by at least the longest window in use (~2–3s or more).
   - *UX effect:* most coherent single view (nothing retroactively changes), at the cost of a visible delay versus true "live."
   - *Best fit:* scenarios prioritizing a clean, non-revising feed over minimum latency — likely a poor fit for the "real-time" goal stated for BallKnowledge, but included for completeness.

4. **Timestamp/sequence-number based reconciliation at the consumer.**
   cv-service emits everything as soon as ready (as in option 1) with strict frame/sequence identifiers; all ordering/reconciliation logic lives in the downstream consumer (e.g., `data_ingestion` or the frontend), not in cv-service.
   - *Latency:* same as option 1 at the producer; consumer-side latency depends on how that service chooses to reconcile.
   - *UX effect:* pushes complexity to whichever service renders the final timeline; cv-service stays simple and consistent with its existing "emit events, don't own downstream state" design (§3.3, FR-010).
   - *Best fit:* consistent with the already-decided architecture where `data_ingestion` owns event routing/ordering guarantees (`kafka.md`: per-match ordering via partition key, `schema_version` field) — likely the most consistent option with existing system design, but requires the downstream consumer to actually implement reconciliation logic, which is not yet designed either.

**Open question (Unknown):** which of these — or a combination — is chosen. This section documents options and trade-offs surfaced during requirements analysis; it is not a design decision, and no option above should be treated as selected.

---

## 5. API / External Interface Requirements

No `API.md` exists for cv-service (unlike `data_service/API.md`); this section is the current source of truth pending one.

All four endpoints are now implemented (`api/main.py`, `docs/architecture/STREAMING_ARCHITECTURE.md` §3-4):

| Endpoint | Purpose | Auth | Success | Failure | Status |
|---|---|---|---|---|---|
| `POST /streams` | Frontend submits a broadcast stream source for validation and processing | `X-CV-Service-Key` header | `201 {"stream_id", "state": "streaming"}` | `422` (malformed/undecodable/no-usable-video), `424` (unreachable), `409` (already active) | Implemented, verified via `api/validate_api.py` (11/11 checks) |
| `POST /streams/{stream_id}/stop` | Frontend stops the active stream/pipeline | `X-CV-Service-Key` header | `200 {"stream_id", "state": "idle"}` | `404` (no such active stream, including "nothing active") | Implemented, verified |
| `GET /streams/{stream_id}/events` (WS) | Live per-frame detection event feed for the active stream, consumed by `data_ingestion`'s `CVPipelineSource` | `?api_key=` query param (same key as REST) | Upgrades, then streams `{"type": "fast"\|"appearance", ...}` JSON messages (raw detections — `CVPipelineSource` performs the translation into `data_ingestion`'s `EventType`/`payload` taxonomy on the consumer side, see `docs/architecture/STREAMING_ARCHITECTURE.md` §4.2, §5.1) | Close code `4401` (bad/missing key), `4404` (no such active stream), `4409` (a consumer is already attached — single-consumer only) | Implemented, verified via `api/validate_ws.py` (9/9 checks) and `data_ingestion`'s `validate_cv_pipeline_source.py` (7/7 checks) |
| `GET /health` | Not part of the original FR-001/003 requirements — added for operability (readiness + current stream state) | none | `200 {"status", "stream_state"}` | — | Implemented, verified |

`source_type` accepts `rtmp`/`rtsp`/`hls`/`file`. `file` routes through `VideoFileFrameSource`; `rtmp`/`rtsp`/`hls` now route through `LiveBroadcastFrameSource` (`docs/architecture/STREAMING_ARCHITECTURE.md` §8 step 8) — real reconnect-on-drop with exponential backoff and a `disconnect_reason` distinguishing a genuine connection loss from normal completion. This logic is verified only via synthetic failure injection (`gsr_pipeline/validate_live_broadcast.py`, mocking `cv2.VideoCapture`) — no real RTMP/RTSP/HLS feed exists in this dev environment, so behavior against an actual live broadcast source (as opposed to the reconnect state machine itself) remains unverified. Treat `rtmp`/`rtsp`/`hls` support as logic-complete but not field-tested.

The WS event feed itself ships the pipeline's raw internal per-frame "fast"/"appearance" event shape (position/role, then jersey/team as it becomes available), not the originally-envisioned `event_type`/`EventType`/`payload` envelope matching `data_ingestion`'s taxonomy 1:1 — but that translation is no longer undone: `CVPipelineSource` (`data_ingestion/src/data_ingestion/cv_pipeline_source.py`) now performs it on the consumer side, converting the raw feed into `PLAYER_POSITION`/`BALL_POSITION`/`PLAYER_IDENTIFIED` `DataEvent`s. `POSSESSION`/`TURNOVER` remain genuinely unwired for live streams (`possession.py` still only runs in cv-service's batch path) — confirmed as a real functional gap, not a translation gap, and not blocking anything documented here.

**Compatibility constraints:** None yet — no prior API version exists to remain compatible with.

**Architectural question — resolved:** how cv-service's *output* reaches `data_ingestion` was raised as an open question earlier in this document's history:
- **Originally planned (Confirmed, `PIPELINE_EVOLUTION.md` §1):** in-process integration — `data_ingestion` imports `gsr_pipeline` as a library, no network boundary.
- **Decided and implemented:** the network-streaming path. cv-service runs as its own process with an HTTP+WS API (§this section); `data_ingestion`'s `CVPipelineSource(BaseDataSource)` is an HTTP+WS *client* of it, not a library consumer — confirming the reasoning in the original plan (cv-service's GPU-heavy dependencies, `torch`/`ultralytics`/`easyocr`, stay isolated from `data_ingestion`). `CVPipelineSource` still publishes into the same in-process pub/sub → `KafkaEventPublisher` path every other source uses; only the cv-service boundary itself is now a network call instead of a function call.
- Verified end-to-end against a real running cv-service server:
  `data_ingestion/src/data_ingestion/validate_cv_pipeline_source.py`, 7/7 checks passing. See `docs/architecture/STREAMING_ARCHITECTURE.md` §5.

---

## 6. Data Requirements

No database exists in cv-service today (Confirmed — `AGENTS.md`: "No web framework, no database, no message queue, no Docker"). Data requirements below describe the target event-emission behavior, not a schema to be implemented inside this repo.

- **Event envelope**: cv-service's emitted events are expected to conform to `data_ingestion`'s existing `DataEvent` envelope and `EventType` taxonomy (Confirmed reference — `data_ingestion/src/data_ingestion/custom_types.py`, `kafka.md`), including a `schema_version` field and per-match ordering key (Confirmed — `PIPELINE_EVOLUTION.md` §4). Every new event type must be mapped into `TOPIC_GROUPS` in `kafka_publisher.py`, which raises at import time if any `EventType` is left unmapped (Confirmed — hard requirement per `PIPELINE_EVOLUTION.md` §4).
- **Required fields per event type**: role, position (pixel + pitch-relative where calibrated), track ID, team, jersey number, event-specific payload (e.g., action type for classifier events). Exact per-type schemas: Unknown — not finalized in `PIPELINE_EVOLUTION.md`, only the event *names* and rough semantics are.
- **Data lifecycle**: `PLAYER_IDENTIFIED` is emitted "on first confirmation or on change, not every frame" — an update/correction model, not append-only per frame (Confirmed — `PIPELINE_EVOLUTION.md` §3.1; **Implemented** — `CVPipelineSource._translate_appearance_event` keeps a per-track `(team, jersey_number)` cache and only emits on an actual change, since cv-service's own WS feed does not enforce this dedup itself). Position events (`PLAYER_POSITION`, `BALL_POSITION`) are per-frame/append-only (**Implemented**, confirming the earlier Inferred status — `CVPipelineSource._translate_fast_event` emits one per detection row per WS "fast" message, which arrives every frame).
- **Retention**: Not specified anywhere for cv-service. `data_ingestion`'s own retention policy (if any) is out of scope for this document.
- **Validation rules**: None specified for event payloads. The one explicit validation-like rule that exists is at the pipeline's internal DataFrame layer (`utils.require_columns()`, Confirmed but an implementation detail, not a product requirement) and is out of scope here.
- **State transitions**: `TURNOVER` requires a prior `POSSESSION` state to compare against (§FR-006). `INTERCEPTION` is emitted as a follow-up/refinement to an already-fired `TURNOVER` (Confirmed — `PIPELINE_EVOLUTION.md` §3.2) — it cannot be emitted standalone.

---

## 7. Business Rules

- A `TURNOVER` event requires a preceding possession-change comparison; it is derived from `POSSESSION`, not an independent detection (Confirmed — §FR-006, `PIPELINE_EVOLUTION.md` §3.1).
- `INTERCEPTION` is a sub-type/refinement of an already-emitted `TURNOVER`; it cannot exist without one (Confirmed — `PIPELINE_EVOLUTION.md` §3.2).
- Entity role (player/goalkeeper/referee/ball) is authoritative from the detector only — no downstream stage may reassign it based on position or appearance heuristics (Confirmed — `AGENTS.md` Architecture Rule 7; explicitly, a prior spatial-heuristic goalkeeper reassignment was identified as a bug and removed).
- Team assignment: only "player"-classified entities are used to *fit* the two team clusters — referees are never assigned a team (Confirmed — `AGENTS.md`, `docs/architecture/GSR_PIPELINE.md` §2). Goalkeepers are excluded from the fit for the same reason (kit doesn't resemble either outfield team) but are assigned a side afterward via nearest-centroid distance against the fitted clusters (Fixed — `AGENTS.md` "Resolved since initial documentation"; previously goalkeepers were left with `team = None` indefinitely, which was a genuine gap for possession/turnover payloads).
- Only one stream may be actively registered/processing at a time in v1 (Stated — product owner); a second registration attempt while one is active must be handled per a rule that is not yet defined (Unknown — see FR-001 constraints).
- A stream must pass validation (reachability, decodability, "producing visible video") before it is accepted and processing begins (Stated — product owner; §FR-001, FR-002).
- Raw video/pixel data must never cross the boundary out of cv-service to downstream BallKnowledge services (Confirmed — `PIPELINE_EVOLUTION.md` §1; §FR-010).

---

## 8. Error Handling Requirements

| Condition | Expected observable behavior | Status |
|---|---|---|
| Invalid/malformed stream source input | Rejected at registration with a `4xx` status | Stated (product owner); exact code Unknown |
| Unreachable stream source | Rejected with a distinguishable `4xx`, not a hang/timeout with no response | Stated (product owner) |
| Undecodable / unsupported format | Rejected with a distinguishable `4xx` | Stated (product owner) |
| Stream reachable but producing no usable video (black/frozen frame) | Rejected with a distinguishable `4xx` | Stated (product owner) |
| Registration attempted while a stream is already active | Rejected; exact status/semantics undecided | Unknown (flagged, §FR-001) |
| Shutdown requested with no active stream | Defined, non-error response expected; exact behavior undecided | Unknown (flagged, §FR-003) |
| Stream disconnects mid-processing (post-registration) | Not specified — no reconnection/retry/failure-event behavior documented | Unknown |
| A pipeline stage errors mid-sequence (e.g., calibration failure, OCR failure) | Not specified for the live/service context. In the current batch pipeline, some per-row failures are caught and logged/skipped rather than aborting the run (Observed — `AGENTS.md` "Error handling"), but this is implementation behavior in the offline CLI, not a documented service-level contract | Observed (batch only) / Unknown (service-level) |
| Downstream event-sink (`data_ingestion`/Kafka) unavailable or backpressured | Not specified for cv-service. `data_ingestion` documents backpressure handling on its own side (Confirmed reference — `PIPELINE_EVOLUTION.md` §1), but how cv-service itself behaves if that sink is unreachable is undocumented | Unknown |
| Duplicate/replayed registration requests (idempotency) | Not specified | Unknown |

---

## 9. Security Requirements

- Authentication is implemented: a static API-key header (`X-CV-Service-Key`, `api/auth.py`), value from the `CV_SERVICE_API_KEY` environment variable. The dependency fails closed (HTTP `500`) if the env var is unset, rather than silently allowing unauthenticated access. No authorization/role model beyond "has the key" exists — appropriate for the current internal service-to-service use case (`docs/architecture/STREAMING_ARCHITECTURE.md` §3.3), not yet designed for any broader exposure.
- No secrets or credentials are present in tracked files in `cv_service/` (Confirmed — `AGENTS.md` Configuration section); the only environment variable found repo-wide (`SOCCERNET_PASSWORD`, used by the offline `video_ingestion` notebook, removed 2026-10-12) is unrelated to the live-service endpoints and out of scope here.
- The one existing deliberate outbound network call (calibration checkpoint download from Zenodo) is documented and treated as an approved exception — new silent outbound network calls are explicitly disallowed without review (Confirmed — `AGENTS.md` "Forbidden / Dangerous Changes").
- Stream-source validation (FR-002) doubles as a basic input-safety gate (rejecting unreachable/malformed sources before they're processed), but this is a functional/reliability requirement, not a documented security control — treat any security framing of it as **Inferred**, not confirmed.

---

## 10. Compatibility Requirements

- **Existing APIs**: None exist yet for the live service; nothing to preserve compatibility with today.
- **Offline pipeline output contract**: The current batch pipeline's output format (SoccerNet-GSR-format `predictions.json`) and the GS-HOTA evaluation formula are treated as a protected public contract for compatibility with official SoccerNet-GSR tooling (Confirmed — `AGENTS.md` Architecture Rule 6, "Forbidden / Dangerous Changes"). This applies to the existing offline/benchmark use of the pipeline; it is explicitly **not** the production output contract, which is planned to be `DataEvent`s into `data_ingestion` instead (Confirmed — `PIPELINE_EVOLUTION.md` §1).
- **Event schema compatibility**: New event types must integrate into `data_ingestion`'s existing `EventType`/envelope/Kafka-routing scheme without requiring changes to that scheme's ordering/delivery guarantees (Confirmed — `PIPELINE_EVOLUTION.md` §4).
- **Detector class mapping**: `yolov8n_gsr.pt`'s 4-class mapping (`player=0, goalkeeper=1, referee=2, ball=3`) must stay in sync with `detect.py` — a compatibility constraint between a committed model artifact and code (Confirmed — `AGENTS.md` Architecture Rule 5, "Forbidden / Dangerous Changes").
- **Supported platforms/versions**: No Python version is pinned anywhere in the repo (Confirmed absence — `AGENTS.md`). No supported OS/deployment target is documented beyond the fact that development currently happens without Docker/CI.

---

## 11. Constraints

- **No API/service framework currently in the repo** — building the endpoints in §5 is greenfield work; there is no existing FastAPI/Flask/etc. scaffolding in `cv_service/` to build on (Confirmed absence — `AGENTS.md`: "No web framework... This is an offline batch-processing research pipeline, not a networked service").
- **No database, message queue, or Docker setup in this repo** — any live-service state (active-stream tracking, etc.) currently has nowhere to live within `cv_service/` (Confirmed absence — `AGENTS.md`).
- **Heavy, GPU-oriented dependency footprint** (`torch`, `ultralytics`, `torchreid`, `easyocr`) was the stated reason the original plan favored in-process integration with `data_ingestion` (`PIPELINE_EVOLUTION.md` §1) — resolved differently in practice: the network-streaming path (§5) was built instead, which isolates these dependencies inside cv-service's own process just as well, without requiring `data_ingestion` to import them.
- **Batch-only algorithmic design of tracklet aggregation, refinement, and (originally) team classification** — these stages, as originally written, require the complete sequence/tracklet before producing output and cannot run incrementally without a genuine redesign (Confirmed — `AGENTS.md` "Batch-only design vs. real-time goal"). Team classification specifically now has a live/incremental counterpart (`LiveTeamAssigner`, §FR-005) — tracklet aggregation and refinement remain genuinely batch-only, with no live equivalent, same as before.
- **`nbjw_calib` is a separately pip-installed vendored third-party package** or its own setup step — not a regular in-repo module (Confirmed — `AGENTS.md` Architecture Rule 4).
- **Committed model artifact dependency**: `yolov8n_gsr.pt` is a fine-tuned, committed weights file with a specific class mapping baked into the code; it cannot be swapped or retrained casually (Confirmed — `AGENTS.md` "Forbidden / Dangerous Changes").
- **Downstream Kafka topic-mapping requirement**: any new event type added to support this service must be mapped in `data_ingestion`'s `TOPIC_GROUPS`, or that module fails at import time — a hard cross-repo coupling constraint (Confirmed — `PIPELINE_EVOLUTION.md` §4). Every `EventType` `CVPipelineSource` emits (`PLAYER_POSITION`, `BALL_POSITION`, `PLAYER_IDENTIFIED`) was already mapped before this work started.
- **cv-service and `data_ingestion` are separate git repositories** (`cv_service` has its own remote, `github.com/SabihAli/ml-project`) — resolved by the network-integration design: `CVPipelineSource` lives inside `data_ingestion` and talks to cv-service purely over HTTP/WS, so there is no shared/glue code straddling both repos (unlike the originally-considered in-process import, which would have required one).

---

## 12. Out of Scope

- Any processing of non-broadcast-video input (e.g., GPS trackers, dedicated tracking-camera rigs, manual event tagging) — the input is explicitly "ONLY the live football broadcast stream" (Stated — product owner). Whether historical/recorded (non-live) broadcast footage is ever a supported input is **Unknown** — not stated either way.
- Serving raw video/pixels to any downstream BallKnowledge component — explicitly disallowed by design (Confirmed — §FR-010, `PIPELINE_EVOLUTION.md` §1).
- Referee-crop stoppage events (foul, card, corner, goal kick) — explicitly deferred past v1 (Confirmed — §FR-008).
- Any UI/visualization rendering — cv-service emits structured events only; presentation is entirely a frontend/downstream concern (Inferred from the "structured data only" boundary, consistent with §FR-010, though not stated in these exact terms anywhere).
- Multi-stream/multi-match concurrent processing — out of scope for v1 specifically (Stated — product owner), though not necessarily out of scope forever; flagged as "Unknown" for future versions rather than permanently excluded.
- GS-HOTA-style offline benchmarking/evaluation as a *production* feature — this remains a development/validation tool for the offline pipeline (Confirmed current use — `AGENTS.md`), not a capability exposed to BallKnowledge end users.
- Direct Kafka production from cv-service — explicitly deferred in the original plan, though now reopened as a question (see §5) — treat as **Unknown**, not confirmed out of scope.

---

## 13. Acceptance Criteria (system-level summary)

- [ ] A frontend client can submit a stream source and receive either `201 Registered` (and processing begins) or a distinguishable `4xx` failure reflecting *why* validation failed (unreachable / undecodable / no usable video).
- [ ] A frontend client can stop an active stream and have all associated processing and resources released.
- [ ] While a stream is active, entity detections (position, role) are produced continuously at per-frame granularity.
- [ ] Entity identity (team, jersey number) converges to a single stable value per tracked entity rather than flickering.
- [ ] Possession and turnover signals are derivable from tracking data alone, without requiring the classifier-based action pipeline.
- [ ] No raw video or pixel data is ever exposed outside cv-service.
- [ ] System logs/traces remain informative under sustained high-frame-rate load without degrading into per-frame spam — achieved via aggregation/anomaly-based observability rather than naive per-event logging (Stated — product owner; design not yet produced).
- [ ] Only one stream is processed at a time in v1; the system does not silently corrupt or mix state from a second concurrent registration attempt.

---

## 14. Requirement Traceability

| Requirement | Evidence |
|---|---|
| FR-001, FR-002, FR-003 (endpoints) | `cv_service/api/main.py`, `api/stream_manager.py`, `api/schemas.py`, `api/auth.py`, `gsr_pipeline/frame_source.py`'s `validate()`; `docs/architecture/STREAMING_ARCHITECTURE.md` §3; verified via `api/validate_api.py` (11/11 checks against a real running server). |
| WS event feed (§5 table, not a numbered FR yet) | `cv_service/api/main.py`'s `stream_events`, `api/event_broadcaster.py`, `api/event_serializer.py`; `docs/architecture/STREAMING_ARCHITECTURE.md` §4; verified via `api/validate_ws.py` (9/9 checks against a real running server + real WS client). |
| `CVPipelineSource` / cv-service → `data_ingestion` integration (§5 architectural question, §6 event taxonomy) | `data_ingestion/src/data_ingestion/cv_pipeline_source.py`; `docs/architecture/STREAMING_ARCHITECTURE.md` §5; verified via `data_ingestion/src/data_ingestion/validate_cv_pipeline_source.py` (7/7 checks against a real running cv-service server). |
| FR-004 (per-frame detection) | `cv_service/gsr_pipeline/detect.py`, `reid.py`, `calibrate.py`; `cv_service/AGENTS.md` "Data flow / Phase 1" |
| FR-005 (tracking/team/jersey identity) | `cv_service/gsr_pipeline/track.py`, `team.py`, `jersey.py`, `tracklet_agg.py`; `cv_service/AGENTS.md` "Data flow / Phase 2"; `cv_service/docs/architecture/PIPELINE_EVOLUTION.md` §3.1 |
| FR-006 (possession/turnover) | `cv_service/gsr_pipeline/possession.py`, `possession_demo.py`, `configs/default.yaml` (`possession` section); `data_ingestion/src/data_ingestion/custom_types.py` + `kafka_publisher.py` (`POSSESSION`/`TURNOVER` enum + topic mapping); `cv_service/docs/architecture/PIPELINE_EVOLUTION.md` §3.1, §5 (build order step 2). **Implemented as an offline batch post-processing step** (validated against `data/outputs/SNGS-021_eval75/predictions.json` and a synthetic scenario) — not yet wired to a live detect+track stream; see FR-005's real-time constraint, which still applies to sourcing `team` for this step. |
| FR-007 (classifier-based actions) | `cv_service/docs/architecture/PIPELINE_EVOLUTION.md` §2, §3.2, §5 (build order step 4) — not yet in code |
| FR-008 (referee-crop events) | `cv_service/docs/architecture/PIPELINE_EVOLUTION.md` §3.3, §5 (build order step 5) — explicitly deferred |
| FR-009 (broadcast graphics OCR) | Stated directly by product owner in this task; no repo evidence |
| FR-010 (structured-event-only boundary) | `cv_service/docs/architecture/PIPELINE_EVOLUTION.md` §1 |
| Data envelope / Kafka mapping (§6) | `data_ingestion/src/data_ingestion/custom_types.py`, `kafka_publisher.py`; `data_ingestion/kafka.md`; `cv_service/docs/architecture/PIPELINE_EVOLUTION.md` §4 |
| Batch-only architectural constraint (§11) | `cv_service/AGENTS.md` "Batch-only design vs. real-time goal" |
| Output-format compatibility (§10) | `cv_service/gsr_pipeline/output.py`, `evaluate.py`; `cv_service/AGENTS.md` Architecture Rule 6 |

---

## Analysis

**1. Files inspected:** `cv_service/README.md`, `cv_service/IMPLEMENTATION.md`, `cv_service/AGENTS.md`, `cv_service/docs/architecture/PIPELINE_EVOLUTION.md`, `cv_service/gsr_pipeline/` directory listing (`pipeline.py`, `detect.py`, `track.py`, `team.py`, `jersey.py`, `calibrate.py`, `tracklet_agg.py`, `refine.py`, `output.py`, `evaluate.py`, `run.py`, `configs/default.yaml` — referenced via AGENTS.md, not individually re-read since AGENTS.md's descriptions were independently verifiable against the repo structure), `cv_service/video_ingestion/` listing (since removed), repository root layout (`data_ingestion/`, `data_service/`, `feature_service/`, `data_inspection/`), `data_ingestion/README.md`, and a targeted grep across `data_ingestion/src/data_ingestion/` for `EventType`/`TOPIC_GROUPS`/`schema_version` usage.

**2. Confirmed requirements discovered:** the event taxonomy and its rationale (§3.1–3.3 of `PIPELINE_EVOLUTION.md`), the "structured data only, no pixels downstream" boundary, the dual short/long window design for classifier events, the fast-partial/slow-refined pattern (`TURNOVER`→`INTERCEPTION`), the Kafka topic-group mapping hard requirement, the batch-only limitation of tracklet/team/refinement stages, and the existing (offline-only) detection/tracking/calibration/role/team/jersey capabilities.

**3. Important observed-but-unconfirmed behaviors:** the offline pipeline's per-row error-suppression pattern (log-and-skip rather than abort) was observed in the batch CLI context; whether this pattern should carry over to the live service's error-handling contract (§8) is not stated and shouldn't be assumed.

**4. Ambiguities or contradictions:**
- The production integration path between cv-service and `data_ingestion` has two candidate designs in tension: the original in-process/library-import plan (`PIPELINE_EVOLUTION.md`) vs. a network-streaming (WebSocket/SSE) alternative raised by the product owner during this task. Neither is confirmed; §5 records both without picking one.
- `GSR_PIPELINE.md`'s (formerly `IMPLEMENTATION.md`) reported GS-HOTA numbers (30.72% mean) and `PIPELINE_EVOLUTION.md`'s single-sequence verification run (69.10% on a 75-frame kickoff segment) are non-comparable figures that could be misread as contradictory if not read carefully (already flagged in `AGENTS.md`; repeated here as a live ambiguity risk for anyone citing pipeline accuracy).

**5. Requirements that could not be determined:** exact HTTP status-code taxonomy for the `4xx` validation failures; request/response schemas for both endpoints; authentication/authorization model; behavior on concurrent registration attempts; behavior on shutdown with no active stream; behavior on mid-stream disconnection or downstream sink unavailability; any numeric performance/latency/throughput target; any retention policy; any deployment/infrastructure plan (Docker, orchestration, scaling).

**6. Assumptions that require human confirmation:**
- That "single active stream" (v1) means outright rejection of a second registration, rather than queuing or replacing — the product owner confirmed *single-stream* but not the rejection mechanics.
- That the synchronization strategy will eventually be one of the four options in §4, or some combination — none has been selected, and this document should not be read as pre-selecting option 2 just because it's the most fleshed-out in existing docs.
- That broadcast-graphics OCR (score/clock) is intended as a first-class event type analogous to the taxonomy in `PIPELINE_EVOLUTION.md`, rather than a separate metadata channel — not specified either way.
