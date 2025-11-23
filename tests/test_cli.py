"""Tests for CLI interface."""

from click.testing import CliRunner

from extract_slides.cli import main


def test_main_help():
    """Test that the main command shows help."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Extract slides from presentations" in result.output


def test_version():
    """Test version display."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_info_command():
    """Test info command."""
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "extract-slides version" in result.output
    assert "0.1.0" in result.output


def test_extract_command_help():
    """Test extract command help."""
    runner = CliRunner()
    result = runner.invoke(main, ["extract", "--help"])
    assert result.exit_code == 0
    assert "Extract slides from a presentation file" in result.output


def test_extract_command_with_nonexistent_file():
    """Test extract command with non-existent file."""
    runner = CliRunner()
    result = runner.invoke(main, ["extract", "nonexistent.pptx"])
    assert result.exit_code != 0


def test_extract_command_with_temp_file():
    """Test extract command with a temporary file."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a dummy file
        with open("test.pptx", "w") as f:
            f.write("dummy content")

        result = runner.invoke(main, ["extract", "test.pptx", "-o", "slides"])
        assert result.exit_code == 0
        assert "Successfully extracted" in result.output


def test_extract_command_verbose():
    """Test extract command with verbose flag."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a dummy file
        with open("test.pptx", "w") as f:
            f.write("dummy content")

        result = runner.invoke(main, ["extract", "test.pptx", "-v"])
        assert result.exit_code == 0
        assert "Extracting slides from" in result.output


def test_extract_command_custom_format():
    """Test extract command with custom format."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a dummy file
        with open("test.pptx", "w") as f:
            f.write("dummy content")

        result = runner.invoke(main, ["extract", "test.pptx", "-f", "jpg"])
        assert result.exit_code == 0
