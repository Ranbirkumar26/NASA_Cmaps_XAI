from __future__ import annotations

import numpy as np

from cmapss_xai.xai.project import (
    sensor_occlusion_attribution,
    temporal_occlusion_attribution,
)


class SumModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return x.sum(axis=1)


def test_sensor_occlusion_attribution_matches_removed_feature_mass() -> None:
    samples = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    baseline = np.zeros((2, 2))
    attr = sensor_occlusion_attribution(SumModel(), samples, ["a", "b"], baseline)
    np.testing.assert_allclose(attr, np.array([[4.0, 6.0], [12.0, 14.0]]))


def test_temporal_occlusion_attribution_matches_removed_time_mass() -> None:
    samples = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    baseline = np.zeros((2, 2))
    attr = temporal_occlusion_attribution(SumModel(), samples, baseline)
    np.testing.assert_allclose(attr, np.array([[3.0, 7.0]]))
