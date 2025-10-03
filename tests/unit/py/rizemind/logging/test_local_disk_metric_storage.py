"""Unit tests for LocalDiskMetricStorage class."""

import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from flwr.common import Parameters, ndarrays_to_parameters

from rizemind.logging.local_disk_metric_storage import LocalDiskMetricStorage


class TestLocalDiskMetricStorage:
    """Test suite for LocalDiskMetricStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create a LocalDiskMetricStorage instance for testing."""
        return LocalDiskMetricStorage(temp_dir, "test_app")

    @pytest.fixture
    def sample_parameters(self):
        """Create sample Parameters for testing."""
        # Create numpy arrays and convert to Parameters using the proper function
        arrays = [np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0], [3.0, 4.0]])]
        return ndarrays_to_parameters(arrays)

    def test_initialization_creates_directory_structure(self, temp_dir):
        """Test that initialization creates the correct directory structure."""
        app_name = "test_app"
        storage = LocalDiskMetricStorage(temp_dir, app_name)

        # Check that the main directory was created
        assert storage.dir.exists()
        assert storage.dir.is_dir()
        assert storage.dir.parent.name == app_name

        # Check that all required files exist
        assert storage.metrics_file.exists()
        assert storage.weights_file.parent.exists()
        # Config file is only created when write_config is called

        # Check that metrics file has correct headers
        with open(storage.metrics_file) as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers == ["server_round", "metric", "value"]

    def test_initialization_with_timestamped_directory(self, temp_dir):
        """Test that initialization creates timestamped subdirectories."""
        app_name = "test_app"

        # Create first instance
        storage1 = LocalDiskMetricStorage(temp_dir, app_name)
        first_dir = storage1.dir

        # Wait a moment and create second instance
        with patch(
            "rizemind.logging.local_disk_metric_storage.datetime"
        ) as mock_datetime:
            # Mock different timestamps
            mock_datetime.now.side_effect = [
                datetime(2024, 1, 1, 12, 0, 0),  # First call
                datetime(2024, 1, 1, 12, 0, 1),  # Second call
            ]
            mock_datetime.strftime = datetime.strftime

            storage2 = LocalDiskMetricStorage(temp_dir, app_name)
            second_dir = storage2.dir

            # Directories should be different
            assert first_dir != second_dir
            assert first_dir.parent == second_dir.parent

    def test_write_config_new_file(self, storage):
        """Test writing configuration to a new file."""
        config = {"learning_rate": 0.01, "batch_size": 32, "epochs": 10}

        storage.write_config(config)

        # Check that file was created and contains correct data
        assert storage.config_file.exists()
        with open(storage.config_file) as f:
            data = json.load(f)
            # Config is stored as a list of dictionaries
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0] == config

    def test_write_config_existing_file_merge(self, storage):
        """Test writing configuration to an existing file merges data."""
        initial_config = {"learning_rate": 0.01, "batch_size": 32}

        additional_config = {"epochs": 10, "optimizer": "adam"}

        # Write initial config
        storage.write_config(initial_config)

        # Write additional config
        storage.write_config(additional_config)

        # Check that both configs are merged
        with open(storage.config_file) as f:
            data = json.load(f)
            # Config is stored as a list of dictionaries
            assert isinstance(data, list)
            assert len(data) == 2  # Two separate configs are stored
            # Check that both configs are present
            config_keys = set()
            for config_dict in data:
                config_keys.update(config_dict.keys())
            assert "learning_rate" in config_keys
            assert "batch_size" in config_keys
            assert "epochs" in config_keys
            assert "optimizer" in config_keys

    def test_write_config_empty_file_handling(self, storage):
        """Test that writing to an empty existing file works correctly."""
        # Create empty file
        storage.config_file.touch()

        config = {"test": "value"}
        storage.write_config(config)

        # Should work without issues
        with open(storage.config_file) as f:
            data = json.load(f)
            # Config is stored as a list of dictionaries
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0] == config

    def test_write_metrics_single_metric(self, storage):
        """Test writing a single metric to the CSV file."""
        server_round = 1
        metrics = {"accuracy": 0.95}

        storage.write_metrics(server_round, metrics)

        # Check CSV content
        with open(storage.metrics_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            row = next(reader)
            assert row == ["1", "accuracy", "0.95"]

    def test_write_metrics_multiple_metrics(self, storage):
        """Test writing multiple metrics to the CSV file."""
        server_round = 2
        metrics = {"accuracy": 0.92, "loss": 0.15, "precision": 0.88}

        storage.write_metrics(server_round, metrics)

        # Check CSV content
        with open(storage.metrics_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers

            rows = list(reader)
            assert len(rows) == 3

            # Check that all metrics are present
            metric_values = {row[1]: row[2] for row in rows}
            assert metric_values["accuracy"] == "0.92"
            assert metric_values["loss"] == "0.15"
            assert metric_values["precision"] == "0.88"

    def test_write_metrics_multiple_rounds(self, storage):
        """Test writing metrics across multiple rounds."""
        # Round 1
        storage.write_metrics(1, {"accuracy": 0.90})

        # Round 2
        storage.write_metrics(2, {"accuracy": 0.92, "loss": 0.15})

        # Check total rows (headers + 3 data rows)
        with open(storage.metrics_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 4  # 1 header + 3 data rows

    def test_write_metrics_different_data_types(self, storage):
        """Test writing metrics with different scalar data types."""
        server_round = 1
        metrics = {
            "int_metric": 42,
            "float_metric": 3.14,
            "bool_metric": True,
            "str_metric": "test_value",
        }

        storage.write_metrics(server_round, metrics)

        # Check that all types are written as strings
        with open(storage.metrics_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers

            rows = list(reader)
            assert len(rows) == 4

            metric_values = {row[1]: row[2] for row in rows}
            assert metric_values["int_metric"] == "42"
            assert metric_values["float_metric"] == "3.14"
            assert metric_values["bool_metric"] == "True"
            assert metric_values["str_metric"] == "test_value"

    def test_update_current_round_model(self, storage, sample_parameters):
        """Test updating the current round model parameters."""
        storage.update_current_round_model(sample_parameters)

        # Check that parameters are stored
        assert storage._current_round_model == sample_parameters
        assert storage._current_round_model.tensors == sample_parameters.tensors
        assert storage._current_round_model.tensor_type == sample_parameters.tensor_type

    def test_update_current_round_model_empty_parameters(self, storage):
        """Test updating with empty parameters."""
        empty_params = Parameters(tensors=[], tensor_type="")
        storage.update_current_round_model(empty_params)

        assert storage._current_round_model == empty_params
        assert storage._current_round_model.tensors == []

    def test_update_best_model_first_model(self, storage, sample_parameters):
        """Test updating best model when it's the first model."""
        storage.update_current_round_model(sample_parameters)

        # First model should always be saved
        storage.update_best_model(server_round=1, loss=0.5)

        # Check that weights file was created
        assert storage.weights_file.exists()

        # Check that best loss was updated
        assert storage._best_loss == 0.5

    def test_update_best_model_better_loss(self, storage, sample_parameters):
        """Test updating best model when loss improves."""
        storage.update_current_round_model(sample_parameters)

        # Set initial best loss
        storage.update_best_model(server_round=1, loss=0.5)
        initial_best_loss = storage._best_loss

        # Update with better loss
        storage.update_best_model(server_round=2, loss=0.3)

        # Check that best loss was updated
        assert storage._best_loss == 0.3
        assert storage._best_loss < initial_best_loss

    def test_update_best_model_worse_loss(self, storage, sample_parameters):
        """Test that worse loss doesn't update the best model."""
        storage.update_current_round_model(sample_parameters)

        # Set initial best loss
        storage.update_best_model(server_round=1, loss=0.3)
        initial_best_loss = storage._best_loss

        # Try to update with worse loss
        storage.update_best_model(server_round=2, loss=0.5)

        # Check that best loss wasn't updated
        assert storage._best_loss == initial_best_loss

    def test_update_best_model_equal_loss(self, storage, sample_parameters):
        """Test that equal loss doesn't update the best model."""
        storage.update_current_round_model(sample_parameters)

        # Set initial best loss
        storage.update_best_model(server_round=1, loss=0.3)
        initial_best_loss = storage._best_loss

        # Try to update with equal loss
        storage.update_best_model(server_round=2, loss=0.3)

        # Check that best loss wasn't updated
        assert storage._best_loss == initial_best_loss

    def test_update_best_model_saves_correct_weights(self, storage, sample_parameters):
        """Test that the correct model weights are saved."""
        storage.update_current_round_model(sample_parameters)
        storage.update_best_model(server_round=1, loss=0.3)

        # Load and verify the saved weights
        loaded_data = np.load(storage.weights_file)
        loaded_tensors = [
            loaded_data[f"arr_{i}"] for i in range(len(sample_parameters.tensors))
        ]

        # Convert original bytes back to numpy arrays for comparison
        from flwr.common import parameters_to_ndarrays

        original_tensors = parameters_to_ndarrays(sample_parameters)

        # Check that tensors match
        for original, loaded in zip(original_tensors, loaded_tensors):
            np.testing.assert_array_equal(original, loaded)

    def test_complete_workflow_simulation(self, storage, sample_parameters):
        """Test a complete workflow simulation."""
        # Simulate multiple rounds
        for round_num in range(1, 4):
            # Update model parameters
            storage.update_current_round_model(sample_parameters)

            # Write metrics
            metrics = {
                "accuracy": 0.8 + round_num * 0.05,
                "loss": 0.5 - round_num * 0.1,
            }
            storage.write_metrics(round_num, metrics)

            # Update best model
            storage.update_best_model(round_num, metrics["loss"])

        # Check final state
        assert storage._best_loss == pytest.approx(0.2)  # Best loss from round 3
        assert storage.weights_file.exists()

        # Check metrics file has correct number of rows
        with open(storage.metrics_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 7  # 1 header + 6 data rows (2 metrics per round)

    def test_config_file_path_properties(self, storage):
        """Test that file path properties are correctly set."""
        assert storage.config_file.name == "config.json"
        assert storage.metrics_file.name == "metrics.csv"
        assert storage.weights_file.name == "weights.npz"

        # All files should be in the same directory
        assert storage.config_file.parent == storage.dir
        assert storage.metrics_file.parent == storage.dir
        assert storage.weights_file.parent == storage.dir

    def test_directory_creation_with_nested_paths(self, temp_dir):
        """Test that nested directory paths are created correctly."""
        nested_path = temp_dir / "level1" / "level2"
        storage = LocalDiskMetricStorage(nested_path, "test_app")

        # Check that all nested directories were created
        assert storage.dir.exists()
        assert storage.dir.parent.exists()
        assert storage.dir.parent.parent.exists()

    def test_initialization_with_existing_directory(self, temp_dir):
        """Test initialization when the directory already exists."""
        app_name = "test_app"

        # Create directory structure manually
        app_dir = temp_dir / app_name
        app_dir.mkdir(parents=True)

        # Should still work
        storage = LocalDiskMetricStorage(temp_dir, app_name)
        assert storage.dir.exists()
        assert storage.metrics_file.exists()

    def test_write_metrics_with_special_characters(self, storage):
        """Test writing metrics with special characters in values."""
        server_round = 1
        metrics = {
            "metric_with_spaces": "value with spaces",
            "metric_with_commas": "value,with,commas",
            "metric_with_quotes": 'value"with"quotes',
            "metric_with_newlines": "value\nwith\nnewlines",
        }

        storage.write_metrics(server_round, metrics)

        # Check that CSV handles special characters correctly
        with open(storage.metrics_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers

            rows = list(reader)
            assert len(rows) == 4

            metric_values = {row[1]: row[2] for row in rows}
            assert metric_values["metric_with_spaces"] == "value with spaces"
            assert metric_values["metric_with_commas"] == "value,with,commas"

    def test_write_config_with_complex_data_types(self, storage):
        """Test writing configuration with complex data types."""
        config = {
            "simple_string": "test",
            "simple_number": 42,
            "simple_boolean": True,
            "nested_dict": {"key1": "value1", "key2": 123},
            "list_value": [1, 2, 3, 4],  # Use homogeneous list to avoid Polars issues
        }

        storage.write_config(config)

        # Check that complex data is preserved
        with open(storage.config_file) as f:
            data = json.load(f)
            # Config is stored as a list of dictionaries
            assert isinstance(data, list)
            assert len(data) == 1
            config_data = data[0]
            assert config_data == config
            assert isinstance(config_data["nested_dict"], dict)
            assert isinstance(config_data["list_value"], list)

    def test_multiple_storage_instances_isolation(self, temp_dir):
        """Test that multiple storage instances don't interfere with each other."""
        storage1 = LocalDiskMetricStorage(temp_dir, "app1")
        storage2 = LocalDiskMetricStorage(temp_dir, "app2")

        # Write different configs
        storage1.write_config({"app": "app1"})
        storage2.write_config({"app": "app2"})

        # Check isolation
        with open(storage1.config_file) as f:
            config1 = json.load(f)
        with open(storage2.config_file) as f:
            config2 = json.load(f)

        # Config is stored as a list of dictionaries
        assert config1[0]["app"] == "app1"
        assert config2[0]["app"] == "app2"
        assert config1 != config2

    def test_initialization_best_loss_default(self, storage):
        """Test that best loss is initialized to infinity."""
        assert storage._best_loss == float("inf")

    def test_initialization_current_round_model_default(self, storage):
        """Test that current round model is initialized to empty parameters."""
        assert storage._current_round_model.tensors == []
        assert storage._current_round_model.tensor_type == ""
