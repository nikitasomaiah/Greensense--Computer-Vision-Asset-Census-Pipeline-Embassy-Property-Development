"""
tests/test_evaluate.py — Pytest suite for evaluate.py accuracy benchmark module
----------------------------------------------------------------------------------
"""

import os
import pytest
from evaluate import EVAL_FIXTURES, generate_eval_images, run_evaluation


def test_eval_fixtures_format():
    """Verify evaluation ground truth fixtures structure."""
    assert len(EVAL_FIXTURES) >= 10
    for fix in EVAL_FIXTURES:
        assert "filename" in fix
        assert "scene_type" in fix
        assert "ground_truth" in fix
        assert isinstance(fix["ground_truth"], dict)


def test_generate_eval_images(tmp_path):
    """Test generating synthetic evaluation image fixtures."""
    out_dir = str(tmp_path / "eval_test_dir")
    generate_eval_images(out_dir)
    assert os.path.exists(out_dir)
    for fix in EVAL_FIXTURES:
        assert os.path.exists(os.path.join(out_dir, fix["filename"]))
