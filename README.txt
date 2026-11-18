CAMPUS SAFETY MONITORING - FIRE/SMOKE TRAINING PACK

Copy the five Python scripts into your project root:
- prepare_dfire_dataset.py
- train_fire_smoke.py
- validate_fire_smoke.py
- test_campus_negative.py
- evaluate_fire_smoke_full_dataset.py

PowerShell commands (from C:\\Users\\nalla\\Desktop\\Campus-Safety-Monitoring):

.\\.venv\\Scripts\\Activate.ps1
python -m pip install -U ultralytics opencv-python pyyaml

Check ARCHIVE at the top of prepare_dfire_dataset.py. It defaults to:
Path.home() / "Downloads" / "DFire-Downloads" / "D-Fire.zip"

Then run, in order:
python prepare_dfire_dataset.py
python train_fire_smoke.py
python validate_fire_smoke.py
python test_campus_negative.py
python evaluate_fire_smoke_full_dataset.py

Expected model:
runs/dfire_training/yolo11n_fire_smoke/weights/best.pt

Training uses CPU for compatibility and can take several hours. If you have a
working NVIDIA CUDA installation, change device="cpu" to device=0 in the
training and validation scripts.

The preparation script creates a deterministic 90/10 train/validation split
from D-Fire's train partition and preserves its official test partition.
Class mapping: 0=smoke, 1=fire.

The video evaluator assumes FP=negative and VP=positive naming. Verify that
against the downloaded video dataset's documentation before treating metrics
as final. Its output is video-level metrics, not object-detection mAP.

IMPORTANT: Keep the existing FireSmokeProcessor integration disabled in
dashboard.py while training/testing. Do not re-enable fire/smoke alerts until
the new model passes the normal-campus negative-video test and the FP/VP
evaluation has been reviewed. Keep datasets/, runs/, reports/, and large
weights out of Git unless you intentionally want to version them.
