# Joint parameter-information study

Frozen computation: `7f9073d18b14a329264c5508204231c72cc5262f`.
Protocol: [`joint-uncertainty-protocol.json`](../../../experiments/joint-uncertainty-protocol.json).
Methods: [`joint-uncertainty.md`](../../../docs/joint-uncertainty.md).
Results: [`joint-results.md`](../../../docs/joint-results.md).

`full/` preserves 22 compressed per-replicate tables, rate/paired/geometry
summaries, exact selected certificates, computational provenance, a data/certificate
audit and an independent rational moment-bound audit, three figures in
PNG/SVG/PDF, and the research report. Raw computational
evidence is never overwritten. The audit reruns 114 shifted datasets and 342
complete certificates, not all numerical decisions. The separate bounds audit checks
678 usable scale cutoffs from saved snapshots; it imports no research implementation
and does not re-verify determinant polynomials.

Install from a source checkout with Python >=3.10:

```bash
python -m pip install -e '.[figures,dev]'
# Rebuild the presentation from saved and audited evidence:
python scripts/joint_report.py --output results/research/joint/full
# Recompute with a new empty directory:
python scripts/joint_uncertainty.py --profile full --workers 4 --output results/research/joint/reproduced
python scripts/verify_joint_uncertainty.py --output results/research/joint/reproduced
python scripts/verify_joint_bounds.py --output results/research/joint/reproduced
python scripts/joint_report.py --output results/research/joint/reproduced
# Package the saved report for Pages:
python scripts/build_joint_site.py
python scripts/build_joint_site.py --check
```

Worker count affects execution order only. Data seeds, record ordering and
compressed table bytes are deterministic; elapsed time and environment metadata
may differ. `--profile quick` checks the pipeline, not calibration or power.

The original environment's editable distribution metadata reads 0.3.0; the
research code is identified by the frozen commit and source hashes. The new
public interface belongs to software 0.4.0. Provenance is retained verbatim.

Coverage and strong FWER concern the documented ideal common Gaussian AR model;
machine certificates concern supplied binary64 values. G8 violates the common
parameter assumption. G10 contains duplicate columns and intentionally loses
joint shape information. Negative results and coverage failures remain saved.
