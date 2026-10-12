"""Private preparation metadata must survive ordinary wheel installation."""

from importlib.metadata import distribution


def test_core_installed_notice_and_upload_guard():
    package = distribution("bt-agent")
    assert "Private :: Do Not Upload" in package.metadata.get_all("Classifier", [])
    assert {"LICENSE", "LICENSE-SCOPE.md", "NOTICE.md"} <= set(
        package.metadata.get_all("License-File", [])
    )
    assert package.metadata["License"].startswith("Apache-2.0 for project-authored")
    assert "data exclusions" in package.metadata["License"]
    from hashlib import sha256

    def installed(name):
        return next(
            package.locate_file(p)
            for p in package.files
            if p.name == name and any(part.endswith(".dist-info") for part in p.parts)
        )

    assert (
        sha256(installed("LICENSE").read_bytes()).hexdigest()
        == "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )
    scope = installed("LICENSE-SCOPE.md").read_text()
    assert "No new data license is granted" in scope
    assert "Third-party code retains" in scope
    notices = [
        path
        for path in package.files
        if path.name == "NOTICE.md" and any(part.endswith(".dist-info") for part in path.parts)
    ]
    assert len(notices) == 1 and package.locate_file(notices[0]).is_file()
