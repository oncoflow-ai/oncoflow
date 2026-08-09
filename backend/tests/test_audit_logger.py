import json
from pathlib import Path
from unittest.mock import patch
from app.core.audit import log_audit_event
import app.main
import logging
import app.core.config

def test_audit_logger_json_format(tmp_path: Path):
    original_settings = app.core.config.get_settings()
    class TestSettings:
        def __getattr__(self, name):
            if name == "storage_root":
                return str(tmp_path)
            return getattr(original_settings, name)
            
    with patch("app.main.get_settings", return_value=TestSettings()):
        # Clear existing handlers to force re-creation
        audit_logger = logging.getLogger("oncoflow.audit")
        audit_logger.handlers.clear()
        
        # Re-configure logging to trigger the file creation in tmp_path
        app.main.configure_logging()
        
        # Fire an audit event
        log_audit_event(
            action="TEST_ACTION",
            resource_id="test-1234",
            actor="pytest_user",
            details={"key": "value"}
        )
        
        # Check that audit.log was created
        audit_log_path = tmp_path / "audit.log"
        assert audit_log_path.exists()
        
        # Read the JSON log
        content = audit_log_path.read_text().strip()
        log_entries = content.split("\n")
        
        # The last log entry should be our test event
        last_log = json.loads(log_entries[-1])
        
        assert last_log["action"] == "TEST_ACTION"
        assert last_log["resource_id"] == "test-1234"
        assert last_log["actor"] == "pytest_user"
        assert last_log["details"] == {"key": "value"}
        assert "asctime" in last_log
        assert "levelname" in last_log
        assert last_log["levelname"] == "INFO"
