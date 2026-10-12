"""
api/ — cv-service's HTTP (and, later, WebSocket) layer.

Implements docs/architecture/STREAMING_ARCHITECTURE.md §3 (stream registration/shutdown)
and, eventually, §4 (the WebSocket event feed -- not yet built, build-order
step 7). Greenfield: no web framework existed in this repo before this
package (see AGENTS.md).
"""
