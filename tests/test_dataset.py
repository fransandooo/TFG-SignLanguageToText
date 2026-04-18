"""Tests for data loading and transforms."""

import numpy as np
import pytest

from slr.data.transforms import SkeletalAugmentation, pad_or_truncate


class TestPadOrTruncate:
    def test_exact_length(self):
        lm = np.random.randn(64, 21, 3).astype(np.float32)
        result = pad_or_truncate(lm, 64)
        assert result.shape == (64, 21, 3)
        np.testing.assert_array_equal(result, lm)

    def test_pad_short(self):
        lm = np.ones((30, 21, 3), dtype=np.float32)
        result = pad_or_truncate(lm, 64)
        assert result.shape == (64, 21, 3)
        np.testing.assert_array_equal(result[:30], lm)
        np.testing.assert_array_equal(result[30:], 0)

    def test_truncate_long(self):
        lm = np.random.randn(120, 21, 3).astype(np.float32)
        result = pad_or_truncate(lm, 64)
        assert result.shape == (64, 21, 3)

    def test_single_frame(self):
        lm = np.ones((1, 21, 3), dtype=np.float32)
        result = pad_or_truncate(lm, 64)
        assert result.shape == (64, 21, 3)
        np.testing.assert_array_equal(result[0], lm[0])


class TestSkeletalAugmentation:
    def test_output_shape(self):
        aug = SkeletalAugmentation(p=1.0)
        lm = np.random.randn(64, 21, 3).astype(np.float32)
        result = aug(lm)
        assert result.shape == lm.shape

    def test_does_not_modify_original(self):
        aug = SkeletalAugmentation(p=1.0)
        lm = np.random.randn(64, 21, 3).astype(np.float32)
        original = lm.copy()
        _ = aug(lm)
        np.testing.assert_array_equal(lm, original)

    def test_no_op_when_p_zero(self):
        aug = SkeletalAugmentation(p=0.0)
        lm = np.random.randn(64, 21, 3).astype(np.float32)
        result = aug(lm)
        np.testing.assert_array_equal(result, lm)
