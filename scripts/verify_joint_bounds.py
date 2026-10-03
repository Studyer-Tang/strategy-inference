"""Independently verify saved Wilks moment bounds without rerunning experiments.

Uses Fraction Beta-integral identities and integer powers; imports no research
implementation. This verifies critical values and scale schemas, not raw-data
regeneration or the correctness of the saved determinant polynomials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from functools import lru_cache
from math import prod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fraction(value: object) -> Fraction:
    _require(isinstance(value, str), "Saved cutoffs must be rational strings.")
    number = Fraction(value)
    _require(str(number) == value, "Saved cutoff is not a canonical Fraction.")
    return number


def _sha(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


@lru_cache(maxsize=256)
def _beta_moment(dimension: int, q: int, s: int, order: int) -> Fraction:
    """Product of B(a+h,b)/B(a,b), using integer b=q/2.

    Gamma(a+b)/Gamma(a)=(a)_b gives each factor
    (a)_b/(a+h)_b. Twice a=s-i+1 may be odd; no Gamma
    approximation or special-function implementation is used.
    """
    b = q // 2
    numerator = prod(s - i + 1 + 2 * v for i in range(1, dimension + 1) for v in range(b))
    denominator = prod(
        s - i + 1 + 2 * v + 2 * order for i in range(1, dimension + 1) for v in range(b)
    )
    _require(
        order >= 0 or s - dimension + 1 + 2 * order > 0,
        "Negative moment is outside its integrability domain.",
    )
    _require(denominator > 0, "Beta-integral product has a nonpositive denominator.")
    return Fraction(numerator, denominator)


def _positive_orders(n: int) -> set[int]:
    ceiling = 1
    while ceiling < 4 * n:
        ceiling *= 2
    values = {1, 2, 3, 4}
    value = 4
    while value <= ceiling:
        values.add(value)
        if 3 * value // 2 <= ceiling:
            values.add(3 * value // 2)
        value *= 2
    return values


@lru_cache(maxsize=128)
def _check_cutoffs(
    dimension: int,
    q: int,
    s: int,
    beta: Fraction,
    bits: int,
    lower: Fraction,
    upper: Fraction,
    lower_h: int,
    upper_h: int,
) -> None:
    denominator = 1 << bits
    step = Fraction(1, denominator)
    _require(0 <= lower < upper <= 1, "Invalid ordered Wilks cutoffs.")
    _require(
        denominator % lower.denominator == 0 and denominator % upper.denominator == 0,
        "Cutoffs are not on the declared dyadic grid.",
    )
    _require(
        0 <= lower_h <= min(64, (s - dimension) // 2), "Selected negative moment is inadmissible."
    )
    if lower_h:
        moment = _beta_moment(dimension, q, s, -lower_h)
        target = beta / (2 * moment)
        _require(
            lower**lower_h <= target < (lower + step) ** lower_h,
            "Lower cutoff is not an exact outward dyadic root.",
        )
        _require(
            moment * lower**lower_h <= beta / 2,
            "Lower Chernoff miss probability exceeds its budget.",
        )
    else:
        _require(lower == 0, "Zero lower moment must have the trivial zero cutoff.")
    if upper_h:
        _require(
            upper_h in _positive_orders(q + s + 1),
            "Selected positive moment is outside the prespecified search.",
        )
        _require(upper < 1, "A nontrivial upper moment must improve the trivial upper cutoff.")
        moment = _beta_moment(dimension, q, s, upper_h)
        target = 2 * moment / beta
        _require(
            (upper - step) ** upper_h < target <= upper**upper_h,
            "Upper cutoff is not an exact outward dyadic root.",
        )
        _require(
            moment <= (beta / 2) * upper**upper_h,
            "Upper Chernoff miss probability exceeds its budget.",
        )
    else:
        _require(upper == 1, "Zero upper moment must use Lambda<=1.")


def verify(output: Path) -> dict[str, object]:
    metadata_bytes = (output / "metadata.json").read_bytes()
    certificate_bytes = (output / "certificates.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    certificates = json.loads(certificate_bytes)
    _require(
        isinstance(metadata, dict) and metadata.get("status") == "complete",
        "A completed computation is required.",
    )
    _require(isinstance(certificates, dict), "Certificates must be a JSON object.")
    _require(
        metadata["output_hashes"]["certificates.json"] == _sha(certificate_bytes),
        "Saved certificate hash differs from the computation manifest.",
    )
    protocol, settings = metadata["protocol"], metadata["settings"]
    _require(protocol["study"] == "joint-uncertainty", "Wrong saved study.")
    procedure = protocol["procedure"]
    lengths = tuple(procedure["block_lengths"])
    _require(
        bool(lengths)
        and len(set(lengths)) == len(lengths)
        and all(_integer(x) and x >= 2 for x in lengths),
        "Invalid prespecified block scales.",
    )
    maximum, bits = procedure["max_dimension"], procedure["critical_bits"]
    _require(
        _integer(maximum) and 1 <= maximum <= 8 and _integer(bits) and 8 <= bits <= 128,
        "Invalid dimension or dyadic precision.",
    )
    beta = Fraction(protocol["beta"])
    _require(
        0 < beta < Fraction(protocol["alpha"]) < Fraction(1, 2), "Invalid shared error budget."
    )
    per_scale = beta / len(lengths)  # Include unusable scales; never reallocate.
    groups = {group["id"]: group for group in protocol["groups"]}
    expected = {}
    for cell in metadata["cells"]:
        phase, group, n = cell["phase"], cell["group"], cell["n"]
        setting = {1: "null_replicates", 2: "power_replicates", 3: "partial_replicates"}[phase]
        _require(n == settings[setting], "Snapshot replicate count differs from settings.")
        deltas = (
            tuple(protocol["power"]["standardized_mean_shifts"])
            if phase == 2
            else ((protocol["partial_null"]["standardized_mean_shift"],) if phase == 3 else (0.0,))
        )
        for replicate in sorted({0, n // 2, n - 1}):
            key = f"{cell['key']}-r{replicate:05d}"
            _require(key not in expected, "Duplicate expected snapshot.")
            expected[key] = group, deltas
    _require(
        set(certificates) == set(expected), "Saved snapshot keys differ from the declared manifest."
    )
    datasets = scales = usable = unusable = singular = lower_bounds = upper_bounds = (
        trivial_upper
    ) = 0
    for key, (group_id, deltas) in expected.items():
        shifts = certificates[key]
        _require(
            isinstance(shifts, list) and tuple(row["delta"] for row in shifts) == deltas,
            "Saved snapshot shifts differ from the protocol.",
        )
        group = groups[group_id]
        for row in shifts:
            datasets += 1
            for method in ("wilks_scalar", "wilks_joint"):
                dimension = 1 if method == "wilks_scalar" else min(group["k"], maximum)
                saved_scales = row[method]["scales"]
                _require(
                    isinstance(saved_scales, list)
                    and tuple(x["block_length"] for x in saved_scales) == lengths,
                    "Saved scale order differs from the protocol.",
                )
                for scale in saved_scales:
                    scales += 1
                    length = scale["block_length"]
                    blocks = (group["n_obs"] - 1) // length
                    blocks = max(0, blocks - int(blocks % 2 == 0))
                    n, q, s = blocks * length, max(0, blocks - 1), blocks * (length - 1)
                    _require(
                        (scale["n_innovations"], scale["low_df"], scale["high_df"]) == (n, q, s),
                        "Saved scale dimensions are incorrect.",
                    )
                    is_usable = blocks >= 3 and s >= dimension + 2
                    _require(
                        type(scale["usable"]) is bool and scale["usable"] == is_usable,
                        "Saved scale eligibility is incorrect.",
                    )
                    lower, upper = map(_fraction, scale["cutoffs"])
                    low_h, high_h = scale["lower_moment"], scale["upper_moment"]
                    _require(
                        _integer(low_h) and _integer(high_h), "Moment orders must be integers."
                    )
                    polynomials = (scale["det_within"], scale["det_total"])
                    _require(
                        all(
                            isinstance(p, list)
                            and 1 <= len(p) <= 2 * dimension + 1
                            and all(_integer(c) for c in p)
                            for p in polynomials
                        ),
                        "Invalid determinant coefficient schema.",
                    )
                    zero_within, zero_total = (not any(p) for p in polynomials)
                    fallback = scale["singular_fallback"]
                    _require(type(fallback) is bool, "Singular fallback must be a boolean.")
                    if not is_usable:
                        unusable += 1
                        _require(
                            (lower, upper, low_h, high_h, fallback) == (0, 1, 0, 0, False)
                            and zero_within
                            and zero_total,
                            "An unusable scale must contribute a full-set fallback.",
                        )
                        continue
                    usable += 1
                    _require(
                        fallback == zero_total and (not zero_total or zero_within),
                        "Singular fallback does not match zero determinant evidence.",
                    )
                    singular += int(fallback)
                    _check_cutoffs(dimension, q, s, per_scale, bits, lower, upper, low_h, high_h)
                    lower_bounds += int(low_h > 0)
                    upper_bounds += int(high_h > 0)
                    trivial_upper += int(high_h == 0)
    # Bind only claims actually checked by this independent script.
    _require(
        (output / "metadata.json").read_bytes() == metadata_bytes
        and (output / "certificates.json").read_bytes() == certificate_bytes,
        "Saved evidence changed during verification.",
    )
    return {
        "status": "passed",
        "scope": "Saved Wilks scale dimensions, dyadic roots and exact Chernoff probability bounds; no Monte Carlo rerun or raw-summary verification.",
        "computation_revision": metadata["git_revision"],
        "metadata_sha256": _sha(metadata_bytes),
        "certificates_sha256": _sha(certificate_bytes),
        "verifier_source": "scripts/verify_joint_bounds.py",
        "verifier_sha256": _sha(Path(__file__).read_bytes()),
        "snapshot_bases_checked": len(expected),
        "snapshot_datasets_checked": datasets,
        "scales_checked": scales,
        "usable_scales_checked": usable,
        "unusable_scales_checked": unusable,
        "singular_scales_checked": singular,
        "exact_lower_bounds_checked": lower_bounds,
        "exact_upper_bounds_checked": upper_bounds,
        "trivial_upper_bounds_checked": trivial_upper,
        "shared_beta": str(beta),
        "per_scale_beta": str(per_scale),
        "predeclared_scale_count": len(lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/research/joint/full")
    parser.add_argument("--audit", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Compare an existing audit without writing"
    )
    args = parser.parse_args()
    audit = args.audit or args.output / "bounds-audit.json"
    try:
        result = verify(args.output)
        contents = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        if args.check:
            _require(
                audit.read_bytes() == contents,
                "Saved bounds audit differs from this independent verification.",
            )
        else:
            audit.write_bytes(contents)
    except (ValueError, OSError, KeyError, TypeError, ZeroDivisionError) as exc:
        parser.exit(1, f"Joint bounds verification failed: {exc}\n")
    print(f"Passed: {result['usable_scales_checked']} usable Wilks scales; audit {audit.resolve()}")


if __name__ == "__main__":
    main()
