"""Tests for Flower Parameters serialization."""

import numpy as np
import pytest

from flwr.common.typing import Parameters

from rizemind.tee.params import (
    deserialize_fit_res_for_enclave,
    deserialize_parameters,
    serialize_fit_res_for_enclave,
    serialize_parameters,
)


class TestParametersSerialization:
    def test_round_trip_single_tensor(self):
        tensors = [np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()]
        params = Parameters(tensors=tensors, tensor_type="numpy.ndarray")

        data = serialize_parameters(params)
        recovered = deserialize_parameters(data)

        assert recovered.tensor_type == "numpy.ndarray"
        assert len(recovered.tensors) == 1
        assert recovered.tensors[0] == tensors[0]

    def test_round_trip_multiple_tensors(self):
        tensors = [
            np.array([1.0, 2.0], dtype=np.float32).tobytes(),
            np.array([3.0, 4.0, 5.0], dtype=np.float32).tobytes(),
            np.array([6.0], dtype=np.float32).tobytes(),
        ]
        params = Parameters(tensors=tensors, tensor_type="numpy.ndarray")

        data = serialize_parameters(params)
        recovered = deserialize_parameters(data)

        assert recovered.tensor_type == "numpy.ndarray"
        assert len(recovered.tensors) == 3
        for orig, rec in zip(tensors, recovered.tensors):
            assert orig == rec

    def test_empty_tensors(self):
        params = Parameters(tensors=[], tensor_type="empty")

        data = serialize_parameters(params)
        recovered = deserialize_parameters(data)

        assert recovered.tensor_type == "empty"
        assert len(recovered.tensors) == 0

    def test_preserves_tensor_values(self):
        original = np.array([1.5, -2.3, 0.0, 100.0], dtype=np.float32)
        params = Parameters(
            tensors=[original.tobytes()], tensor_type="numpy.ndarray"
        )

        data = serialize_parameters(params)
        recovered = deserialize_parameters(data)

        result = np.frombuffer(recovered.tensors[0], dtype=np.float32)
        np.testing.assert_array_equal(original, result)


class TestFitResSerialization:
    def test_round_trip(self):
        tensors = [np.array([1.0, 2.0], dtype=np.float32).tobytes()]
        params = Parameters(tensors=tensors, tensor_type="numpy.ndarray")

        data = serialize_fit_res_for_enclave(100, params)
        num_examples, recovered_params = deserialize_fit_res_for_enclave(data)

        assert num_examples == 100
        assert recovered_params.tensor_type == "numpy.ndarray"
        assert recovered_params.tensors[0] == tensors[0]
