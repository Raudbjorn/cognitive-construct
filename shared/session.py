from typing import Dict, Any, Optional
import uuid
import time
from .errors import SkillError, ErrorCode

class Session:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.state: Dict[str, Any] = {}

    def touch(self):
        self.last_accessed = time.time()

    def get(self, key: str, default: Any = None) -> Any:
        self.touch()
        return self.state.get(key, default)

    def set(self, key: str, value: Any):
        self.touch()
        self.state[key] = value

    def delete(self, key: str):
        self.touch()
        if key in self.state:
            del self.state[key]

class SessionManager:
    """
    Manages active sessions. In-memory for now, could be backed by Redis/File.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance.sessions = {}
        return cls._instance

    def get_session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id)
        return self.sessions[session_id]

    def create_session(self) -> Session:
        session = Session()
        self.sessions[session.session_id] = session
        return session

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
