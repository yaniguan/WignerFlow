"""Ensure every committed Colab notebook clones a branch that exists on the remote."""

import ast
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/yaniguan/WignerFlow.git"


def extract_notebook_branches(colab_dir: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for nb_path in sorted(colab_dir.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        branches: list[str] = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            tree = ast.parse(src, filename=nb_path.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == "BRANCH":
                            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                            branches.append(node.value.value)
        found[nb_path.name] = branches
    return found


def fetch_remote_branches(remote: str = REPO_URL) -> set[str] | None:
    """Return remote branch names via `git ls-remote --heads`, or None if completely offline."""
    try:
        res = subprocess.run(
            ["git", "ls-remote", "--heads", remote],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    heads = set()
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("refs/heads/"):
            heads.add(parts[1].removeprefix("refs/heads/"))
    return heads


def assert_branches_exist(nb_branches: dict[str, list[str]], remote_heads: set[str] | None):
    assert nb_branches, "No Colab notebooks found under colab/"
    fallback_allowed = {"main", "v1.0-dev"}
    for nb_name, branches in nb_branches.items():
        assert branches, f"{nb_name} has no BRANCH assignment"
        for br in branches:
            if remote_heads is not None:
                assert br in remote_heads, f"{nb_name} points to missing remote branch {br!r} (existing: {sorted(remote_heads)})"
            else:
                assert br in fallback_allowed, f"{nb_name} points to non-persistent branch {br!r}"


def test_committed_colab_notebooks_point_to_existing_remote_branches():
    nb_branches = extract_notebook_branches(ROOT / "colab")
    remote_heads = fetch_remote_branches()
    assert_branches_exist(nb_branches, remote_heads)
    for branches in nb_branches.values():
        assert branches == ["main"]


def test_missing_colab_branch_fails_check():
    with pytest.raises(AssertionError, match="missing remote branch"):
        assert_branches_exist({"stale.ipynb": ["feat/colab-oct29"]}, {"main", "v1.0-dev"})
    with pytest.raises(AssertionError, match="non-persistent branch"):
        assert_branches_exist({"stale.ipynb": ["feat/20ns-stability"]}, None)
