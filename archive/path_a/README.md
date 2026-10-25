# Path A (archived 2026-10-25)

The original pipeline: a 4-class YOLOv8n (`yolov8n_gsr.pt`: player, goalkeeper, referee,
ball) whose class *was* the role, plus KMeans teams (`team.py`) and the live team assigner
(`team_live.py`, `pitch_view_gate.py`).

Replaced by Path B, which is now the only pipeline: the 2-class athlete/ball detector
(`gsr_pipeline/detect_athlete.py`), roles and teams from kit colour
(`gsr_pipeline/athlete_roles.py`, `gsr_pipeline/live_roles.py`) and the c5c ball specialist
(`gsr_pipeline/ball_specialist.py`). See `docs/path_b/plans/IMPLEMENTATION_PLAN.md` §12.2.

Nothing here is imported by the pipeline, the API or the tests. The files are kept as they
were when archived (their imports still say `gsr_pipeline.*`, so they do not run from here).
Path A's one GS-HOTA figure, 30.72% on the full SoccerNet-GSR test set from an older version
of the code, is recorded in `docs/architecture/GSR_PIPELINE.md`.
