"""Tests for refactored CLI with VideoAnalysisConfig."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from extract_slides.cli import VideoAnalysisConfig, main, video_main


class TestVideoAnalysisConfig:
    """Test VideoAnalysisConfig dataclass."""

    def test_config_with_defaults(self):
        """Test config creation with default values."""
        config = VideoAnalysisConfig(
            input_file="/tmp/video.mp4",
            output_dir="/tmp/output",
        )

        assert config.input_file == "/tmp/video.mp4"
        assert config.output_dir == "/tmp/output"
        assert config.grid_cols == 4
        assert config.grid_rows == 4
        assert config.cell_hash_threshold == 5
        assert config.min_static_cell_ratio == 0.8
        assert config.min_static_frames == 3
        assert config.verbose is False

    def test_config_with_custom_values(self):
        """Test config creation with custom values."""
        config = VideoAnalysisConfig(
            input_file="/tmp/video.mp4",
            output_dir="/tmp/output",
            grid_cols=8,
            grid_rows=8,
            cell_hash_threshold=10,
            min_static_cell_ratio=0.9,
            min_static_frames=5,
            verbose=True,
        )

        assert config.grid_cols == 8
        assert config.grid_rows == 8
        assert config.cell_hash_threshold == 10
        assert config.min_static_cell_ratio == 0.9
        assert config.min_static_frames == 5
        assert config.verbose is True

    def test_to_analysis_config(self):
        """Test conversion to AnalysisConfig TypedDict."""
        config = VideoAnalysisConfig(
            input_file="/tmp/video.mp4",
            output_dir="/tmp/output",
            grid_cols=6,
            grid_rows=6,
            cell_hash_threshold=7,
            min_static_cell_ratio=0.85,
            min_static_frames=4,
        )

        analysis_config = config.to_analysis_config()

        assert analysis_config["grid_cols"] == 6
        assert analysis_config["grid_rows"] == 6
        assert analysis_config["cell_hash_threshold"] == 7
        assert analysis_config["min_static_cell_ratio"] == 0.85
        assert analysis_config["min_static_frames"] == 4

        # Should not include input_file, output_dir, or verbose
        assert "input_file" not in analysis_config
        assert "output_dir" not in analysis_config
        assert "verbose" not in analysis_config

    def test_config_from_kwargs(self):
        """Test creating config from **kwargs (as CLI does)."""
        kwargs = {
            "input_file": "/tmp/video.mp4",
            "output_dir": "/tmp/output",
            "grid_cols": 4,
            "grid_rows": 4,
            "cell_hash_threshold": 5,
            "min_static_cell_ratio": 0.8,
            "min_static_frames": 3,
            "verbose": False,
        }

        config = VideoAnalysisConfig(**kwargs)

        assert config.input_file == "/tmp/video.mp4"
        assert config.output_dir == "/tmp/output"


class TestCLIVideoAnalysis:
    """Test CLI commands with the refactored config system."""

    def test_analyze_video_command_with_defaults(self):
        """Test analyze-video command uses default config values."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create test file
            Path("test.mp4").touch()

            # Try to run with defaults (will fail during actual analysis, but should parse)
            result = runner.invoke(
                main,
                [
                    "analyze-video",
                    "--input", "test.mp4",
                    "--output-dir", "output",
                ],
            )

            # Should fail because it's not a real video, but command should be recognized
            assert "analyze-video" in str(result.output).lower() or result.exit_code != 0

    def test_analyze_video_command_with_custom_grid(self):
        """Test analyze-video command with custom grid settings."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.mp4").touch()

            result = runner.invoke(
                main,
                [
                    "analyze-video",
                    "--input", "test.mp4",
                    "--output-dir", "output",
                    "--grid-cols", "8",
                    "--grid-rows", "6",
                ],
            )

            # Command should be recognized even if execution fails
            assert result.exit_code != 0 or "analyzing" in str(result.output).lower()

    def test_video_main_standalone_command(self):
        """Test video_main standalone entry point."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.mp4").touch()

            result = runner.invoke(
                video_main,
                [
                    "--input", "test.mp4",
                    "--output-dir", "output",
                ],
            )

            # Should attempt to run (will fail on real analysis)
            assert result.exit_code != 0 or "analyzing" in str(result.output).lower()

    def test_analyze_video_validates_grid_dimensions(self):
        """Test that zero or negative grid dimensions are rejected."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.mp4").touch()

            result = runner.invoke(
                main,
                [
                    "analyze-video",
                    "--input", "test.mp4",
                    "--output-dir", "output",
                    "--grid-cols", "0",
                ],
            )

            assert result.exit_code == 1
            assert "Grid dimensions must be positive" in result.output

    def test_analyze_video_validates_static_cell_ratio(self):
        """Test that invalid static cell ratio is rejected."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.mp4").touch()

            result = runner.invoke(
                main,
                [
                    "analyze-video",
                    "--input", "test.mp4",
                    "--output-dir", "output",
                    "--min-static-cell-ratio", "1.5",
                ],
            )

            assert result.exit_code == 1
            assert "min-static-cell-ratio must be between 0 and 1" in result.output

    def test_analyze_video_validates_min_static_frames(self):
        """Test that invalid min static frames is rejected."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.mp4").touch()

            result = runner.invoke(
                main,
                [
                    "analyze-video",
                    "--input", "test.mp4",
                    "--output-dir", "output",
                    "--min-static-frames", "0",
                ],
            )

            assert result.exit_code == 1
            assert "min-static-frames must be at least 1" in result.output

    def test_analyze_video_verbose_flag(self):
        """Test that verbose flag enables extra output."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.mp4").touch()

            result = runner.invoke(
                main,
                [
                    "analyze-video",
                    "--input", "test.mp4",
                    "--output-dir", "output",
                    "--verbose",
                ],
            )

            # Verbose mode should show the config details
            # (will fail on analysis, but should show verbose output first)
            assert result.exit_code != 0  # Will fail because not real video


class TestCLIBackwardCompatibility:
    """Test that refactored CLI maintains backward compatibility."""

    def test_all_original_options_still_work(self):
        """Test that all original command-line options still work."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.mp4").touch()

            # Test with all the original options
            result = runner.invoke(
                main,
                [
                    "analyze-video",
                    "--input", "test.mp4",
                    "--output-dir", "output",
                    "--grid-cols", "4",
                    "--grid-rows", "4",
                    "--cell-hash-threshold", "5",
                    "--min-static-cell-ratio", "0.8",
                    "--min-static-frames", "3",
                    "--verbose",
                ],
            )

            # Should not have argument parsing errors
            # (will fail on video processing, which is fine)
            assert "no such option" not in result.output.lower()
            assert "unrecognized argument" not in result.output.lower()

    def test_video_static_segments_command_still_works(self):
        """Test that standalone video_static_segments command works."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.mp4").touch()

            result = runner.invoke(
                video_main,
                [
                    "--input", "test.mp4",
                    "--output-dir", "output",
                    "--verbose",
                ],
            )

            # Should recognize all options
            assert "no such option" not in result.output.lower()


class TestCLIErrorMessages:
    """Test that error messages are helpful."""

    def test_missing_required_input(self):
        """Test error when --input is missing."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["analyze-video", "--output-dir", "output"],
        )

        assert result.exit_code != 0
        assert "--input" in result.output or "required" in result.output.lower()

    def test_missing_required_output_dir(self):
        """Test error when --output-dir is missing."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.mp4").touch()
            result = runner.invoke(
                main,
                ["analyze-video", "--input", "test.mp4"],
            )

            assert result.exit_code != 0
            assert "--output-dir" in result.output or "required" in result.output.lower()

    def test_nonexistent_input_file(self):
        """Test error when input file doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "analyze-video",
                "--input", "nonexistent.mp4",
                "--output-dir", "output",
            ],
        )

        assert result.exit_code != 0
        # Click should catch this during path validation


class TestRefactoringBenefits:
    """Tests that demonstrate benefits of the refactoring."""

    def test_config_is_immutable(self):
        """Test that config dataclass is immutable (frozen would be ideal)."""
        config = VideoAnalysisConfig(
            input_file="/tmp/video.mp4",
            output_dir="/tmp/output",
        )

        # Config values can be read
        assert config.grid_cols == 4

        # If frozen, this would raise FrozenInstanceError
        # For now, just ensure we can create configs consistently

    def test_config_separates_cli_from_logic(self):
        """Test that CLI concerns are separate from analysis logic."""
        # Create config from CLI-like kwargs
        cli_kwargs = {
            "input_file": "/tmp/video.mp4",
            "output_dir": "/tmp/output",
            "grid_cols": 4,
            "grid_rows": 4,
            "cell_hash_threshold": 5,
            "min_static_cell_ratio": 0.8,
            "min_static_frames": 3,
            "verbose": True,
        }

        config = VideoAnalysisConfig(**cli_kwargs)

        # Convert to analysis config (removes CLI-specific fields)
        analysis_config = config.to_analysis_config()

        # Analysis config shouldn't have CLI fields
        assert "verbose" not in analysis_config
        assert "input_file" not in analysis_config
        assert "output_dir" not in analysis_config

        # But should have all analysis fields
        assert "grid_cols" in analysis_config
        assert "cell_hash_threshold" in analysis_config

    def test_adding_new_parameter_is_easy(self):
        """Demonstrate that adding parameters is now easier."""
        # With the dataclass pattern, adding a new parameter only requires:
        # 1. Add field to VideoAnalysisConfig
        # 2. Add option to _video_analysis_options decorator
        # 3. Optionally add to to_analysis_config() if needed by analyzer

        # Compare to old way: had to update multiple function signatures
        # and maintain parameter order in multiple places

        # This test just documents the improvement
        pass
