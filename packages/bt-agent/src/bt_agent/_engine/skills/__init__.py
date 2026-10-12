"""LAM v2 skills package.

Provides skill doc loading for progressive disclosure — the actor
includes brief tool definitions in every call and loads full .md
documentation on demand via the read_skill_docs tool.

Substrate packages register their doc directories at import time via
:func:`register_doc_root`. For example, ``lam_v2_tiamat/__init__.py``
calls ``register_doc_root(Path(__file__).parent / "bt_skills" / "docs")``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

_SKILLS_ROOT = Path(__file__).parent

# Substrate packages register additional doc search paths here.
_extra_doc_roots: List[Path] = []


def register_doc_root(path: Path) -> None:
    """Register a directory containing skill ``.md`` documentation.

    Called by substrate packages at import time so their skill docs are
    discoverable by the actor's ``read_skill_docs`` tool.
    """
    if path not in _extra_doc_roots:
        _extra_doc_roots.append(path)


def _discover_doc_files() -> Dict[str, Path]:
    """Return ``{skill_name: md_path}`` across all registered doc roots.

    Searches two locations:
    1. ``skills/*/docs/`` — framework-level skill docs (if any)
    2. Registered substrate doc roots (via :func:`register_doc_root`)

    First-write-wins on collisions: if two packages ship a doc for the same
    skill name, the first-registered package's doc is kept and the second
    is ignored. Skill names should be unique across packages anyway
    (the registry rejects duplicates).
    """
    docs: Dict[str, Path] = {}
    # Framework-level skill docs (currently none, but supports future additions)
    for docs_dir in sorted(_SKILLS_ROOT.glob("*/docs")):
        if not docs_dir.is_dir():
            continue
        for md_file in sorted(docs_dir.glob("*.md")):
            docs.setdefault(md_file.stem, md_file)
    # Substrate-registered doc roots
    for root in _extra_doc_roots:
        if not root.is_dir():
            continue
        for md_file in sorted(root.glob("*.md")):
            docs.setdefault(md_file.stem, md_file)
    return docs


def list_skill_docs() -> List[str]:
    """Return names of all available skill docs (without .md extension).

    >>> list_skill_docs()
    ['capture', 'image_inject', 'navigate', 'query', 'surround_survey']
    """
    return sorted(_discover_doc_files().keys())


def load_skill_doc(skill_name: str) -> str:
    """Load a single skill's .md documentation by name.

    Returns the file content, or an error message if the doc doesn't exist.
    Used by the actor's read_skill_docs inner loop.
    """
    docs = _discover_doc_files()
    path = docs.get(skill_name)
    if path is None:
        available = ", ".join(sorted(docs.keys()))
        return f"No documentation found for '{skill_name}'. Available: {available}"
    return path.read_text(encoding="utf-8").strip()


def load_skill_docs() -> str:
    """Load ALL skill .md files and return as a single formatted string.

    Each skill's docs are separated by a header line. Used by the planner
    to give the LLM full context when decomposing tasks into milestones.
    """
    sections = []
    for name, path in sorted(_discover_doc_files().items()):
        content = path.read_text(encoding="utf-8").strip()
        sections.append(f"=== SKILL: {name} ===\n{content}")
    return "\n\n".join(sections)
