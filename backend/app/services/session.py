import time
import uuid
from typing import Dict, Any, Optional

# In-memory session store (for production use Redis or similar)
_sessions: Dict[str, Dict[str, Any]] = {}


def create_session() -> Dict[str, Any]:
    """Create a new session"""
    session_id = str(uuid.uuid4())
    session = {
        'session_id': session_id,
        'start_ts': time.time(),
        'id_embedding': None,
        'doc_face_path': None,
        'liveness_passed': False,
        'liveness_instance': None,
        'locked_live_embedding': None,
        'locked_live_image': None,
        'attempts': 0,
        'paused': False,
        'status': 'CREATED'
    }
    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session by ID"""
    return _sessions.get(session_id)


def set_session(session_id: str, updates: Dict[str, Any]) -> None:
    """Update session with new data"""
    if session_id in _sessions:
        _sessions[session_id].update(updates)


def delete_session(session_id: str) -> None:
    """Delete session"""
    if session_id in _sessions:
        del _sessions[session_id]


def cleanup_expired_sessions(timeout: int = 300) -> int:
    """Remove sessions older than timeout (in seconds)"""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s['start_ts'] > timeout]
    for sid in expired:
        del _sessions[sid]
    return len(expired)
