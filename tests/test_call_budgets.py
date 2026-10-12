from unittest.mock import Mock, patch

from bt_agent._engine.agent.lam_agent.actor import _compact_llm, build_history_entry
from bt_agent._engine.model.backend import ModelBackend


def test_compaction_uses_512_on_wire_not_backend_default(tmp_path, monkeypatch):
    from bt_agent._engine.model import backend

    monkeypatch.setattr(backend, "_VLM_LOG_DIR", str(tmp_path))
    model = ModelBackend(
        {"endpoint": "http://offline.invalid/v1", "model_name": "fixture", "max_tokens": 4096}
    )
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": "A report was recorded."}}]}
    entries = build_history_entry(
        "report", {"text": "Observed a mug."}, "Report recorded.", call_id="fixture-1"
    )
    with patch("requests.post", return_value=response) as post:
        result = _compact_llm(entries, model, "Summarize: {actions_text}")
    assert result == "A report was recorded."
    assert post.call_args.kwargs["json"]["max_tokens"] == 512
    assert post.call_args.kwargs["json"]["temperature"] == 0.0
