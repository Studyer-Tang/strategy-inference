"""Small bound certificates; no formal data reads or Monte Carlo computation."""

import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _sha(contents):
    return hashlib.sha256(contents).hexdigest()


@pytest.fixture
def verifier():
    source = PROJECT / "scripts/verify_joint_bounds.py"
    specification = importlib.util.spec_from_file_location("joint_bounds_fixture", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _scale(length, n, q, s, cutoffs, lower_h, upper_h, *, usable=True, singular=False):
    return {
        "block_length": length,
        "n_innovations": n,
        "low_df": q,
        "high_df": s,
        "cutoffs": list(cutoffs),
        "lower_moment": lower_h,
        "upper_moment": upper_h,
        "usable": usable,
        "singular_fallback": singular,
        # This audit checks coefficient schemas and zero fallback evidence;
        # it deliberately makes no claim to regenerate scatter polynomials.
        "det_within": [0] if singular or not usable else [1],
        "det_total": [0] if singular or not usable else [1],
    }


@pytest.fixture
def saved(tmp_path):
    # Frozen rational cuts for T=128, beta=.005/3, 40-bit grid. A singular
    # selected covariance supplies the fallback case without extra datasets.
    scalar = [
        _scale(
            4, 124, 30, 93, ("142818096529/274877906944", "1006775820471/1099511627776"), 30, 128
        ),
        _scale(16, 112, 6, 105, ("832080617919/1099511627776", "1"), 43, 0),
        _scale(64, 64, 0, 63, ("0", "1"), 0, 0, usable=False),
    ]
    joint = [
        _scale(
            4,
            124,
            30,
            93,
            ("43108106629/1099511627776", "3400882869/17179869184"),
            14,
            24,
            singular=True,
        ),
        _scale(
            16,
            112,
            6,
            105,
            ("439038387869/1099511627776", "908154715193/1099511627776"),
            26,
            64,
            singular=True,
        ),
        _scale(64, 64, 0, 63, ("0", "1"), 0, 0, usable=False),
    ]
    certificates = {
        "p1-g01-r00000": [
            {"delta": 0.0, "wilks_scalar": {"scales": scalar}, "wilks_joint": {"scales": joint}}
        ]
    }
    metadata = {
        "status": "complete",
        "git_revision": "0" * 40,
        "settings": {"null_replicates": 1},
        "protocol": {
            "study": "joint-uncertainty",
            "alpha": 0.05,
            "beta": 0.005,
            "procedure": {"block_lengths": [4, 16, 64], "max_dimension": 8, "critical_bits": 40},
            "groups": [{"id": 1, "n_obs": 128, "k": 8}],
        },
        "cells": [{"key": "p1-g01", "phase": 1, "group": 1, "n": 1}],
        "output_hashes": {},
    }

    def write():
        contents = (json.dumps(certificates) + "\n").encode()
        metadata["output_hashes"]["certificates.json"] = _sha(contents)
        (tmp_path / "certificates.json").write_bytes(contents)
        (tmp_path / "metadata.json").write_text(json.dumps(metadata) + "\n")

    write()
    return SimpleNamespace(path=tmp_path, scalar=scalar, joint=joint, write=write)


def test_minimal_certificate_passes_exact_bounds_and_binds_the_evidence(verifier, saved):
    result = verifier.verify(saved.path)
    assert result["status"] == "passed"
    assert result["snapshot_bases_checked"] == result["snapshot_datasets_checked"] == 1
    assert result["scales_checked"] == 6
    assert result["usable_scales_checked"] == result["exact_lower_bounds_checked"] == 4
    assert result["exact_upper_bounds_checked"] == 3
    assert result["trivial_upper_bounds_checked"] == 1
    assert result["unusable_scales_checked"] == result["singular_scales_checked"] == 2
    assert Fraction(result["per_scale_beta"]) == Fraction(0.005) / 3
    assert result["metadata_sha256"] == _sha((saved.path / "metadata.json").read_bytes())
    assert result["certificates_sha256"] == _sha((saved.path / "certificates.json").read_bytes())
    assert result["verifier_sha256"] == _sha(Path(verifier.__file__).read_bytes())


@pytest.mark.parametrize(
    "corruption,error",
    [
        ("inward_lower_root", "Lower cutoff"),
        ("zero_upper_moment", "Zero upper moment"),
        ("wrong_degrees", "Saved scale dimensions"),
        ("missing_singular_fallback", "Singular fallback"),
        ("informative_unusable_scale", "An unusable scale"),
    ],
)
def test_rebound_manifest_cannot_hide_a_false_bound_or_fallback(verifier, saved, corruption, error):
    if corruption == "inward_lower_root":
        saved.scalar[0]["cutoffs"][0] = str(
            Fraction(saved.scalar[0]["cutoffs"][0]) + Fraction(1, 1 << 40)
        )
    elif corruption == "zero_upper_moment":
        saved.scalar[0]["upper_moment"] = 0
    elif corruption == "wrong_degrees":
        saved.scalar[0]["low_df"] = 31
    elif corruption == "missing_singular_fallback":
        saved.joint[0]["singular_fallback"] = False
    else:
        saved.joint[2]["cutoffs"] = ["1/2", "1"]
    # Rebind the outer manifest so rejection must come from the mathematical
    # or structural check, rather than a stale-hash check alone.
    saved.write()
    with pytest.raises(ValueError, match=error):
        verifier.verify(saved.path)
