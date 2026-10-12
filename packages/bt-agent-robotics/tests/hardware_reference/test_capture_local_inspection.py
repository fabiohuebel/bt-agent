from io import BytesIO

from bt_agent.skills import ObjectStore
from bt_agent.skills import SkillContext, SkillStatus
from bt_agent_robotics._hardware.skills.capture import CaptureSkill
from bt_agent_robotics._hardware.skills.query import SceneObject
from PIL import Image


class Model:
    def __init__(self, visible):
        self.visible = visible

    def query_structured(self, *args, **kwargs):
        return {
            "visible": self.visible,
            "bbox_1000": [100, 100, 700, 700] if self.visible else [0, 0, 0, 0],
            "reasoning": "test",
            "acceptable": True,
        }


def context():
    output = BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="JPEG")
    return SkillContext(object_store=ObjectStore(), platform_state={"camera_images": {"frontleft": output.getvalue()}})


def test_default_retains_stored_handle_contract(monkeypatch):
    monkeypatch.delenv("LAM_CAPTURE_LOCAL_INSPECTION", raising=False)
    skill, ctx = CaptureSkill(Model(True)), context()
    assert not skill.precondition(ctx)[0]
    _, error = skill.prepare({"target_object": "screwdriver", "camera_view": "frontleft"}, ctx)
    assert "not in ObjectStore" in error


def test_local_description_needs_no_prior_object(monkeypatch):
    monkeypatch.setenv("LAM_CAPTURE_LOCAL_INSPECTION", "1")
    skill, ctx = CaptureSkill(Model(True)), context()
    assert skill.precondition(ctx)[0]
    params, error = skill.prepare({"target_object": "red_screwdriver", "camera_view": "frontleft"}, ctx)
    assert error is None and params["target_label"] == "red screwdriver"
    assert params["target_handle"] == "" and params["target_source"] == "local_visual_description"
    assert not ctx.object_store.all_objects()  # no invented landmark/coordinates


def test_known_handle_still_uses_original_label(monkeypatch):
    monkeypatch.setenv("LAM_CAPTURE_LOCAL_INSPECTION", "1")
    skill, ctx = CaptureSkill(Model(True)), context()
    ctx.object_store.add(SceneObject(handle="tool_1", label="red tool", object_id="4"))
    params, error = skill.prepare({"target_object": "tool_1", "camera_view": "frontleft"}, ctx)
    assert error is None and params["target_label"] == "red tool" and params["target_handle"] == "tool_1"
    assert "target_source" not in params


def test_local_description_still_requires_camera_and_image(monkeypatch):
    monkeypatch.setenv("LAM_CAPTURE_LOCAL_INSPECTION", "1")
    skill, ctx = CaptureSkill(Model(True)), context()
    _, error = skill.prepare({"target_object": "tool", "camera_view": "unknown"}, ctx)
    assert "Unknown camera" in error
    _, error = skill.prepare({"target_object": "tool", "camera_view": "back"}, ctx)
    assert "No image" in error


def test_visible_local_object_produces_capture_not_landmark(monkeypatch, tmp_path):
    monkeypatch.setenv("LAM_CAPTURE_LOCAL_INSPECTION", "1")
    monkeypatch.setattr("bt_agent_robotics._hardware.skills.capture._CAPTURE_DEBUG_DIR", str(tmp_path))
    skill, ctx = CaptureSkill(Model(True)), context()
    params, error = skill.prepare({"target_object": "tool", "camera_view": "frontleft"}, ctx)
    assert error is None
    handle = skill.start(params, ctx)
    assert handle._completion_event.wait(2)
    assert handle.status == SkillStatus.SUCCEEDED
    skill.publish_results(handle, ctx.object_store)
    objects = ctx.object_store.all_objects()
    assert len(objects) == 1 and objects[0].description == "tool"
    assert not hasattr(objects[0], "viewpoint_position")


def test_invisible_local_object_still_fails(monkeypatch):
    monkeypatch.setenv("LAM_CAPTURE_LOCAL_INSPECTION", "1")
    skill, ctx = CaptureSkill(Model(False)), context()
    params, _ = skill.prepare({"target_object": "tool", "camera_view": "frontleft"}, ctx)
    handle = skill.start(params, ctx)
    assert handle._completion_event.wait(2)
    assert handle.status == SkillStatus.FAILED
    assert not ctx.object_store.all_objects()
