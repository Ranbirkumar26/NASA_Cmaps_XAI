from __future__ import annotations

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SETTING_COLUMNS = [f"setting_{idx}" for idx in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{idx}" for idx in range(1, 22)]
ID_COLUMNS = ["unit_id", "cycle"]
RAW_COLUMNS = [*ID_COLUMNS, *SETTING_COLUMNS, *SENSOR_COLUMNS]
DERIVED_TARGET_COLUMNS = ["rul", "rul_capped"]

SUBSET_DESCRIPTIONS = {
    "FD001": {
        "train_trajectories": 100,
        "test_trajectories": 100,
        "conditions": "ONE (Sea Level)",
        "fault_modes": "ONE (HPC Degradation)",
    },
    "FD002": {
        "train_trajectories": 260,
        "test_trajectories": 259,
        "conditions": "SIX",
        "fault_modes": "ONE (HPC Degradation)",
    },
    "FD003": {
        "train_trajectories": 100,
        "test_trajectories": 100,
        "conditions": "ONE (Sea Level)",
        "fault_modes": "TWO (HPC Degradation, Fan Degradation)",
    },
    "FD004": {
        "train_trajectories": 248,
        "test_trajectories": 249,
        "conditions": "SIX",
        "fault_modes": "TWO (HPC Degradation, Fan Degradation)",
    },
}
