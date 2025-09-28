#!/usr/bin/env python3
"""
Tests for graph2viewer.py module.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_IO_ERROR,
    EXIT_SUCCESS,
)
from viz.graph2viewer import (
    generate_html,
    load_graph_data,
    main,
    minify_json_data,
    save_html,
    setup_logging,
)


@pytest.mark.viz
class TestSetupLogging:
    """Test logging setup."""

    def test_setup_logging_creates_logger(self, tmp_path):
        """Test that setup_logging creates and returns a logger."""
        log_file = tmp_path / "test.log"

        with patch("viz.graph2viewer.logging.basicConfig") as mock_config:
            logger = setup_logging(log_file)

            # Check logger was created
            assert logger is not None
            mock_config.assert_called_once()
            # Check log directory was created
            assert log_file.parent.exists()


@pytest.mark.viz
class TestLoadGraphData:
    """Test data loading functionality."""

    def test_load_production_data_success(self, tmp_path):
        """Test successful loading of production data."""
        # Create test data files
        data_dir = tmp_path / "data" / "out"
        data_dir.mkdir(parents=True)

        graph_data = {
            "nodes": [{"id": "1"}, {"id": "2"}],
            "edges": [{"source": "1", "target": "2"}],
        }
        concepts_data = {"concepts": [{"id": "c1", "term": "Test"}]}

        graph_path = data_dir / "LearningChunkGraph_wow.json"
        concepts_path = data_dir / "ConceptDictionary_wow.json"

        with open(graph_path, "w") as f:
            json.dump(graph_data, f)
        with open(concepts_path, "w") as f:
            json.dump(concepts_data, f)

        logger = MagicMock()

        # Load data
        loaded_graph, loaded_concepts = load_graph_data(data_dir, logger, test_mode=False)

        # Verify
        assert loaded_graph == graph_data
        assert loaded_concepts == concepts_data
        logger.info.assert_any_call("Loaded concepts: 1 concepts")
        logger.info.assert_any_call("Loaded production graph: 2 nodes, 1 edges")

    def test_load_test_data_success(self, tmp_path):
        """Test successful loading of test data."""
        # Create test data files
        test_dir = tmp_path / "data" / "test"
        test_dir.mkdir(parents=True)

        graph_data = {"nodes": [{"id": "1"}], "edges": []}
        concepts_data = {"concepts": []}

        graph_path = test_dir / "tiny_html_data.json"
        concepts_path = test_dir / "tiny_html_concepts.json"

        with open(graph_path, "w") as f:
            json.dump(graph_data, f)
        with open(concepts_path, "w") as f:
            json.dump(concepts_data, f)

        logger = MagicMock()
        data_dir = tmp_path / "data" / "out"

        # Load data in test mode
        loaded_graph, loaded_concepts = load_graph_data(data_dir, logger, test_mode=True)

        # Verify
        assert loaded_graph == graph_data
        assert loaded_concepts == concepts_data
        logger.info.assert_any_call("Loaded test graph: 1 nodes, 0 edges")

    def test_missing_graph_file(self, tmp_path):
        """Test handling of missing graph file."""
        data_dir = tmp_path / "data" / "out"
        logger = MagicMock()

        with pytest.raises(SystemExit) as excinfo:
            load_graph_data(data_dir, logger, test_mode=False)

        assert excinfo.value.code == EXIT_INPUT_ERROR
        logger.error.assert_called()

    def test_missing_concepts_test_mode(self, tmp_path):
        """Test handling of missing concepts file in test mode - should use stub."""
        # Create only graph file
        test_dir = tmp_path / "data" / "test"
        test_dir.mkdir(parents=True)

        graph_data = {"nodes": [], "edges": []}
        graph_path = test_dir / "tiny_html_data.json"

        with open(graph_path, "w") as f:
            json.dump(graph_data, f)

        logger = MagicMock()
        data_dir = tmp_path / "data" / "out"

        # Load data in test mode
        loaded_graph, loaded_concepts = load_graph_data(data_dir, logger, test_mode=True)

        # Verify stub was used
        assert loaded_graph == graph_data
        assert loaded_concepts["concepts"] == []
        assert "_meta" in loaded_concepts
        logger.info.assert_any_call("Test concept dictionary not found, using empty stub")

    def test_invalid_json(self, tmp_path):
        """Test handling of invalid JSON."""
        data_dir = tmp_path / "data" / "out"
        data_dir.mkdir(parents=True)

        graph_path = data_dir / "LearningChunkGraph_wow.json"
        with open(graph_path, "w") as f:
            f.write("invalid json{")

        logger = MagicMock()

        with pytest.raises(SystemExit) as excinfo:
            load_graph_data(data_dir, logger, test_mode=False)

        assert excinfo.value.code == EXIT_INPUT_ERROR
        # Check that error was logged 
        assert logger.error.called


@pytest.mark.viz
class TestMinifyJsonData:
    """Test JSON minification."""

    def test_minify_enabled(self):
        """Test JSON minification when enabled."""
        data = {"key": "value", "nested": {"a": 1}}
        result = minify_json_data(data, minify=True)
        
        # Should be compact
        assert result == '{"key":"value","nested":{"a":1}}'

    def test_minify_disabled(self):
        """Test JSON formatting when minification disabled."""
        data = {"key": "value"}
        result = minify_json_data(data, minify=False)
        
        # Should have formatting
        assert "\n" in result
        assert "  " in result  # indentation

    def test_unicode_preserved(self):
        """Test that Unicode is preserved."""
        data = {"text": "Привет мир! 你好世界"}
        result = minify_json_data(data, minify=True)
        
        # Unicode should not be escaped
        assert "Привет" in result
        assert "你好" in result
        assert "\\u" not in result  # No Unicode escaping


@pytest.mark.viz
class TestGenerateHtml:
    """Test HTML generation."""

    def test_generate_html_production(self, tmp_path):
        """Test HTML generation in production mode."""
        # Setup
        viz_dir = tmp_path / "viz"
        template_dir = viz_dir / "templates" / "viewer"
        template_dir.mkdir(parents=True)
        
        # Create minimal template
        template_content = """
        <!DOCTYPE html>
        <html>
        <head><title>{{ title }}</title></head>
        <body>
            <div>Nodes: {{ node_count }}</div>
            <div>Edges: {{ edge_count }}</div>
            <script>
                window.graphData = {{ graph_data_json|safe }};
            </script>
        </body>
        </html>
        """
        
        with open(template_dir / "index.html", "w") as f:
            f.write(template_content)
        
        with open(template_dir / "viewer_styles.css", "w") as f:
            f.write("body { margin: 0; }")
        
        # Create test modules
        static_dir = viz_dir / "static" / "viewer"
        static_dir.mkdir(parents=True)
        
        for module in ["viewer_core.js", "node_explorer.js", "edge_inspector.js",
                      "navigation_history.js", "formatters.js", "search_filter.js"]:
            with open(static_dir / module, "w") as f:
                f.write(f"// {module}")
        
        graph_data = {
            "nodes": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
            "edges": [{"source": "1", "target": "2"}],
            "_meta": {"title": "Test Graph"}
        }
        concepts_data = {"concepts": []}
        config = {
            "graph2html": {
                "minify_json": True,
                "embed_libraries": True
            },
            "text_formatting": {
                "enable_markdown": True
            }
        }
        
        logger = MagicMock()
        
        # Generate HTML
        html = generate_html(graph_data, concepts_data, config, viz_dir, logger, test_mode=False)
        
        # Verify
        assert "Test Graph" in html
        assert "Nodes: 3" in html
        assert "Edges: 1" in html
        assert "window.graphData" in html
        logger.info.assert_any_call("HTML generated successfully")

    def test_generate_html_with_missing_modules(self, tmp_path):
        """Test HTML generation when some modules are missing."""
        # Setup minimal environment
        viz_dir = tmp_path / "viz"
        template_dir = viz_dir / "templates" / "viewer"
        template_dir.mkdir(parents=True)
        
        # Create minimal template
        with open(template_dir / "index.html", "w") as f:
            f.write("<html><body>{{ title }}</body></html>")
        
        graph_data = {"nodes": [], "edges": []}
        concepts_data = {"concepts": []}
        config = {"graph2html": {}}
        
        logger = MagicMock()
        
        # Generate HTML (modules missing)
        html = generate_html(graph_data, concepts_data, config, viz_dir, logger)
        
        # Should succeed with warnings
        assert html is not None
        # Check for warnings about missing modules
        warning_calls = [call[0][0] for call in logger.warning.call_args_list]
        assert any("viewer_core.js not found" in str(call) for call in warning_calls)


@pytest.mark.viz
class TestSaveHtml:
    """Test HTML saving."""

    def test_save_html_success(self, tmp_path):
        """Test successful HTML saving."""
        output_path = tmp_path / "output.html"
        html = "<html><body>Test</body></html>"
        logger = MagicMock()
        
        save_html(html, output_path, logger)
        
        # Verify file was created
        assert output_path.exists()
        assert output_path.read_text() == html
        logger.info.assert_any_call(f"HTML saved to: {output_path}")

    def test_save_html_creates_directory(self, tmp_path):
        """Test that save_html creates parent directory if needed."""
        output_path = tmp_path / "subdir" / "output.html"
        html = "<html></html>"
        logger = MagicMock()
        
        save_html(html, output_path, logger)
        
        assert output_path.exists()
        assert output_path.parent.exists()


@pytest.mark.viz
class TestMain:
    """Test main entry point."""

    @patch("viz.graph2viewer.load_config")
    @patch("viz.graph2viewer.load_graph_data")
    @patch("viz.graph2viewer.generate_html")
    @patch("viz.graph2viewer.save_html")
    @patch("viz.graph2viewer.setup_console_encoding")
    def test_main_production_mode(
        self, mock_encoding, mock_save, mock_generate, mock_load_data, mock_config
    ):
        """Test main function in production mode."""
        # Setup mocks
        mock_config.return_value = {"graph2html": {}}
        mock_load_data.return_value = (
            {"nodes": [], "edges": []},
            {"concepts": []}
        )
        mock_generate.return_value = "<html></html>"
        
        # Run main with no arguments (production mode)
        with patch("sys.argv", ["graph2viewer.py"]):
            result = main()
        
        assert result == EXIT_SUCCESS
        mock_load_data.assert_called_once()
        assert mock_load_data.call_args[1]["test_mode"] is False
        mock_save.assert_called_once()

    @patch("viz.graph2viewer.load_config")
    @patch("viz.graph2viewer.load_graph_data")
    @patch("viz.graph2viewer.generate_html")
    @patch("viz.graph2viewer.save_html")
    @patch("viz.graph2viewer.setup_console_encoding")
    def test_main_test_mode(
        self, mock_encoding, mock_save, mock_generate, mock_load_data, mock_config
    ):
        """Test main function in test mode."""
        # Setup mocks
        mock_config.return_value = {"graph2html": {}}
        mock_load_data.return_value = (
            {"nodes": [], "edges": []},
            {"concepts": []}
        )
        mock_generate.return_value = "<html></html>"
        
        # Run main with --test flag
        with patch("sys.argv", ["graph2viewer.py", "--test"]):
            result = main()
        
        assert result == EXIT_SUCCESS
        mock_load_data.assert_called_once()
        assert mock_load_data.call_args[1]["test_mode"] is True

    @patch("viz.graph2viewer.load_config")
    @patch("viz.graph2viewer.setup_console_encoding")
    def test_main_config_error(self, mock_encoding, mock_config):
        """Test main function with config error."""
        mock_config.side_effect = Exception("Config error")
        
        with patch("sys.argv", ["graph2viewer.py"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        
        assert excinfo.value.code == EXIT_CONFIG_ERROR