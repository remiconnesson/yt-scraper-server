"""Tests for extraction logic."""

import tempfile
from pathlib import Path

import pytest

from slides_extractor.extract_slides.extractor import extract_slides


def test_extract_slides_creates_output_dir():
    """Test that extract_slides creates output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "test.pptx"
        input_file.write_text("dummy content")
        output_dir = Path(tmpdir) / "output"

        result = extract_slides(str(input_file), str(output_dir), "png")

        assert output_dir.exists()
        assert result["output_dir"] == str(output_dir)


def test_extract_slides_with_nonexistent_file():
    """Test that extract_slides raises error for non-existent file."""
    with pytest.raises(FileNotFoundError):
        extract_slides("nonexistent.pptx", "output", "png")


def test_extract_slides_with_invalid_format():
    """Test that extract_slides raises error for invalid format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "test.pptx"
        input_file.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported format"):
            extract_slides(str(input_file), "output", "invalid")


def test_extract_slides_supported_formats():
    """Test that all supported formats work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "test.pptx"
        input_file.write_text("dummy content")

        for fmt in ["png", "jpg", "pdf"]:
            result = extract_slides(str(input_file), str(Path(tmpdir) / "output"), fmt)
            assert result["format"] == fmt


def test_extract_slides_verbose_mode():
    """Test extract_slides with verbose mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "test.pptx"
        input_file.write_text("dummy content")

        # Should not raise any errors
        result = extract_slides(
            str(input_file), str(Path(tmpdir) / "output"), "png", verbose=True
        )
        assert result["count"] >= 0


def test_extract_slides_result_structure():
    """Test that extract_slides returns correct structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "test.pptx"
        input_file.write_text("dummy content")

        result = extract_slides(str(input_file), str(Path(tmpdir) / "output"), "png")

        assert "count" in result
        assert "output_dir" in result
        assert "format" in result
        assert isinstance(result["count"], int)
        assert isinstance(result["output_dir"], str)
        assert isinstance(result["format"], str)
