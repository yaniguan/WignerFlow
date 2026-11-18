"""Verify that environment.yml, pyproject.toml, and CI workflow agree on e3nn and editable install."""

from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _extract_pyproject_e3nn(text: str) -> str:
    match = re.search(r'"(e3nn[^"]+)"', text)
    assert match is not None, "e3nn dependency not found in pyproject.toml"
    return match.group(1)


def test_e3nn_pin_and_ci_editable_install_match_pyproject():
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    expected_spec = _extract_pyproject_e3nn(pyproject_text)
    assert expected_spec == "e3nn>=0.4.4,<0.6"

    env_cfg = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
    pip_blocks = [item["pip"] for item in env_cfg["dependencies"] if isinstance(item, dict) and "pip" in item]
    assert len(pip_blocks) == 1
    assert expected_spec in pip_blocks[0], f"environment.yml pip section {pip_blocks[0]!r} != {expected_spec!r}"

    ci_cfg = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))
    job = ci_cfg["jobs"]["pytest-cpu"]
    matrix_versions = job["strategy"]["matrix"]["python-version"]
    assert "3.10" in matrix_versions and "3.11" in matrix_versions

    run_scripts = "\n".join(step.get("run", "") for step in job["steps"])
    assert "pip install -e" in run_scripts
    assert "e3nn==" not in run_scripts
