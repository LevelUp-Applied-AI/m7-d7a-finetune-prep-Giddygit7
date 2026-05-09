# Module 7 Week A — Drill: Fine-Tuning Prep

Implement four functions in `drill.py`: `make_dataset`, `tokenize_dataset`, `make_training_args`, `compute_metrics`. The drill exercises the mechanical preparation steps the lab assumes; no actual training.

Full instructions: see the **Core Skills Drill 7A guide** linked in TalentLMS.

## Quick start

```bash
# On Linux, install the CPU torch wheel first to avoid the ~2GB CUDA build:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
git checkout -b drill-7a-finetune-prep
# implement drill.py
pytest tests/ -v
```

(macOS, including Apple Silicon, ships a CPU/MPS wheel by default — no extra step needed.)

## Submission

Open a PR from `drill-7a-finetune-prep` into `main`. Paste the PR URL into TalentLMS → Module 7 → Core Skills Drill 7A.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
