#!/usr/bin/env python3
"""
Tests for viz.graph2html module.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from viz.graph2html import (
    CDN_URLS,
    LIBRARY_ORDER,
    generate_html,
    generate_script_tags,
    generate_style_tags,
    get_library_content,
    load_graph_data,
    load_vendor_content,
    minify_html_content,
    minify_json_data,
    save_html,
    setup_logging,
)


@pytest.mark.viz
class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_creates_directory(self):
        """Test that logging setup creates log directory."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("logging.basicConfig"):
                with patch("logging.FileHandler"):
                    with patch("logging.StreamHandler"):
                        log_file = Path("/test/logs/test.log")
                        logger = setup_logging(log_file)
                        
                        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                        assert logger is not None


@pytest.mark.viz
class TestGetLibraryContent:
    """Tests for get_library_content function - critical CDN fallback logic."""

    def test_get_library_content_local_success(self):
        """Test loading library from local file."""
        mock_content = "console.log('library');"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=mock_content)):
                logger = MagicMock()
                
                result = get_library_content(
                    "test.js", 
                    Path("/test/test.js"), 
                    "https://cdn.example.com/test.js",
                    logger
                )
                
                assert result == mock_content
                logger.info.assert_called()
                logger.warning.assert_not_called()

    def test_get_library_content_cdn_fallback(self):
        """Test CDN fallback when local file missing."""
        mock_response = Mock()
        mock_response.text = "/* CDN content */"
        mock_response.raise_for_status = Mock()
        
        with patch("pathlib.Path.exists", return_value=False):
            with patch("requests.get", return_value=mock_response):
                logger = MagicMock()
                
                result = get_library_content(
                    "cytoscape.min.js",
                    Path("/test/cytoscape.min.js"),
                    CDN_URLS["cytoscape.min.js"],
                    logger
                )
                
                assert result == "/* CDN content */"
                logger.warning.assert_called()  # Warning about missing file
                logger.info.assert_called()  # Info about successful download

    def test_get_library_content_no_requests_library(self):
        """Test error when requests library not available."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("requests.get", side_effect=ImportError("No module named 'requests'")):
                logger = MagicMock()
                
                with pytest.raises(RuntimeError) as exc_info:
                    get_library_content(
                        "critical.js",
                        Path("/test/critical.js"),
                        "https://cdn.example.com/critical.js",
                        logger
                    )
                
                assert "requests not available" in str(exc_info.value)
                logger.error.assert_called()

    def test_get_library_content_cdn_download_failure(self):
        """Test error when CDN download fails."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("requests.get", side_effect=Exception("Network error")):
                logger = MagicMock()
                
                with pytest.raises(RuntimeError) as exc_info:
                    get_library_content(
                        "critical.js",
                        Path("/test/critical.js"),
                        "https://cdn.example.com/critical.js",
                        logger
                    )
                
                assert "Cannot load critical library" in str(exc_info.value)

    def test_get_library_content_local_read_error_then_cdn(self):
        """Test CDN fallback when local file exists but cannot be read."""
        mock_response = Mock()
        mock_response.text = "/* CDN fallback */"
        mock_response.raise_for_status = Mock()
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                with patch("requests.get", return_value=mock_response):
                    logger = MagicMock()
                    
                    result = get_library_content(
                        "test.js",
                        Path("/test/test.js"),
                        "https://cdn.example.com/test.js",
                        logger
                    )
                    
                    assert result == "/* CDN fallback */"
                    assert logger.warning.call_count >= 2  # Warning for read error and CDN fallback


@pytest.mark.viz
class TestLoadGraphData:
    """Tests for load_graph_data function."""

    def test_load_graph_data_success(self):
        """Test successful loading of graph and concept data."""
        mock_graph = {
            "nodes": [{"id": "1", "text": "Node 1"}],
            "edges": [{"source": "1", "target": "2"}],
        }
        mock_concepts = {"concepts": [{"term": "concept1", "definition": "def1"}]}

        with patch("builtins.open", mock_open()):
            # Order corrected: concepts loaded first, then graph
            with patch("json.load", side_effect=[mock_concepts, mock_graph]):
                with patch("pathlib.Path.exists", return_value=True):
                    logger = MagicMock()
                    data_dir = Path("/test/dir")

                    graph_data, concepts_data = load_graph_data(data_dir, logger, test_mode=False)

                    assert graph_data == mock_graph
                    assert concepts_data == mock_concepts
                    assert logger.info.call_count >= 2

    def test_load_graph_data_test_mode(self):
        """Test loading test data in test mode."""
        mock_graph = {"nodes": [], "edges": []}
        mock_concepts = {"concepts": []}
        
        with patch("builtins.open", mock_open()):
            # Order corrected for test mode too: concepts first, then graph
            with patch("json.load", side_effect=[mock_concepts, mock_graph]):
                with patch("pathlib.Path.exists", return_value=True):
                    logger = MagicMock()
                    data_dir = Path("/test/dir")
                    
                    graph_data, concepts_data = load_graph_data(data_dir, logger, test_mode=True)
                    
                    # Check that test paths are used
                    assert graph_data == mock_graph
                    assert concepts_data == mock_concepts
                    logger.info.assert_any_call("Loaded test graph: 0 nodes, 0 edges")

    def test_load_graph_data_test_mode_missing_concepts(self):
        """Test empty stub for missing concepts in test mode."""
        mock_graph = {"nodes": [{"id": "1"}], "edges": []}
        
        with patch("builtins.open", mock_open(read_data='{"nodes": [{"id": "1"}], "edges": []}')):
            with patch("json.load", return_value=mock_graph):
                with patch("pathlib.Path.exists", side_effect=[True, False]):  # Graph exists, concepts don't
                    logger = MagicMock()
                    data_dir = Path("/test/dir")
                    
                    graph_data, concepts_data = load_graph_data(data_dir, logger, test_mode=True)
                    
                    assert graph_data == mock_graph
                    assert concepts_data["concepts"] == []
                    assert "_meta" in concepts_data
                    logger.info.assert_any_call("Test concept dictionary not found, using empty stub")

    def test_load_graph_data_file_not_found(self):
        """Test handling of missing files."""
        with patch("pathlib.Path.exists", return_value=False):
            logger = MagicMock()
            data_dir = Path("/test/dir")

            with pytest.raises(SystemExit) as exc_info:
                load_graph_data(data_dir, logger, test_mode=False)

            assert exc_info.value.code == 2  # EXIT_INPUT_ERROR
            logger.error.assert_called()

    def test_load_graph_data_invalid_json(self):
        """Test handling of invalid JSON."""
        with patch("builtins.open", mock_open(read_data="invalid json")):
            with patch("pathlib.Path.exists", return_value=True):
                # JSONDecodeError on first json.load (concepts) is caught in except block
                with patch("json.load", side_effect=json.JSONDecodeError("msg", "doc", 0)):
                    logger = MagicMock()
                    data_dir = Path("/test/dir")

                    with pytest.raises(SystemExit) as exc_info:
                        load_graph_data(data_dir, logger, test_mode=False)

                    # JSONDecodeError caught in except block, exits with IO_ERROR
                    assert exc_info.value.code == 5  # EXIT_IO_ERROR (from except Exception block)
                    logger.error.assert_called()


@pytest.mark.viz
class TestMinifyJsonData:
    """Tests for minify_json_data function."""

    def test_minify_json_enabled(self):
        """Test JSON minification when enabled."""
        data = {"key": "value", "nested": {"item": 123}}
        result = minify_json_data(data, minify=True)

        assert result == '{"key":"value","nested":{"item":123}}'
        assert "\n" not in result
        assert " " not in result.replace('" ', "").replace(' "', "")

    def test_minify_json_disabled(self):
        """Test pretty print when minification disabled."""
        data = {"key": "value"}
        result = minify_json_data(data, minify=False)

        assert "{\n" in result
        assert '"key": "value"' in result

    def test_minify_json_unicode(self):
        """Test handling of Unicode characters."""
        data = {"text": "Тест 测试 テスト"}
        result = minify_json_data(data, minify=True)

        assert "Тест" in result
        assert "测试" in result
        assert "テスト" in result


@pytest.mark.viz
class TestLoadVendorContent:
    """Tests for load_vendor_content function."""

    def test_load_vendor_content_with_ordering(self):
        """Test that vendor files are reordered according to LIBRARY_ORDER."""
        # Files in wrong order
        file_list = [
            "vendor/cytoscape-cose-bilkent.js",  # Should be 4th
            "vendor/cytoscape.min.js",  # Should be 1st
            "vendor/cose-base.js",  # Should be 3rd
            "vendor/layout-base.js",  # Should be 2nd
            "vendor/other.js",  # Should remain at end
        ]
        
        mock_contents = {
            "cytoscape.min.js": "/* cytoscape */",
            "layout-base.js": "/* layout-base */",
            "cose-base.js": "/* cose-base */",
            "cytoscape-cose-bilkent.js": "/* cose-bilkent */",
            "other.js": "/* other */",
        }
        
        def mock_read_side_effect(*args, **kwargs):
            mock = MagicMock()
            # Extract filename from the path being opened
            if args and hasattr(args[0], 'name'):
                filename = Path(args[0].name).name
                mock.read.return_value = mock_contents.get(filename, "")
            return mock
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=mock_read_side_effect):
                logger = MagicMock()
                viz_dir = Path("/viz")
                
                result = load_vendor_content(viz_dir, file_list, logger, test_mode=False)
                
                # Check order in result
                lines = result.split("\n")
                order_indices = {}
                for i, line in enumerate(lines):
                    if "/* vendor/cytoscape.min.js */" in line:
                        order_indices["cytoscape"] = i
                    elif "/* vendor/layout-base.js */" in line:
                        order_indices["layout"] = i
                    elif "/* vendor/cose-base.js */" in line:
                        order_indices["cose"] = i
                    elif "/* vendor/cytoscape-cose-bilkent.js */" in line:
                        order_indices["bilkent"] = i
                    elif "/* vendor/other.js */" in line:
                        order_indices["other"] = i
                
                # Verify correct order
                assert order_indices["cytoscape"] < order_indices["layout"]
                assert order_indices["layout"] < order_indices["cose"]
                assert order_indices["cose"] < order_indices["bilkent"]
                assert order_indices["bilkent"] < order_indices["other"]

    def test_load_vendor_content_test_mode_with_debug(self):
        """Test that debug_helpers.js is added in test mode."""
        file_list = ["vendor/cytoscape.min.js"]
        
        # Simulate that both files exist
        with patch("pathlib.Path.exists", return_value=True):
            # Mock open to return content for both files
            with patch("builtins.open", mock_open(read_data="/* file content */")):
                logger = MagicMock()
                viz_dir = Path("/viz")
                
                result = load_vendor_content(viz_dir, file_list, logger, test_mode=True)
                
                # Check that both vendor file and debug helpers are included
                assert "/* vendor/cytoscape.min.js */" in result
                assert "/* static/debug_helpers.js */" in result
                # Check that files were actually loaded (both have same mock content)
                assert result.count("/* file content */") == 2
                logger.info.assert_any_call("Added debug_helpers.js for test mode")

    def test_load_vendor_content_cdn_fallback_for_critical(self):
        """Test CDN fallback for critical libraries."""
        file_list = ["vendor/cytoscape.min.js"]  # Critical library
        
        with patch("pathlib.Path.exists", return_value=False):
            with patch("viz.graph2html.get_library_content", return_value="/* CDN content */") as mock_get:
                logger = MagicMock()
                viz_dir = Path("/viz")
                
                result = load_vendor_content(viz_dir, file_list, logger, test_mode=False)
                
                mock_get.assert_called_once_with(
                    "cytoscape.min.js",
                    viz_dir / "vendor/cytoscape.min.js",
                    CDN_URLS["cytoscape.min.js"],
                    logger
                )
                assert "/* vendor/cytoscape.min.js */" in result
                assert "/* CDN content */" in result

    def test_load_vendor_content_skip_missing_non_critical(self):
        """Test that missing non-critical files are skipped."""
        file_list = [
            "vendor/cytoscape.min.js",  # Critical
            "vendor/custom-plugin.js",  # Non-critical (not in CDN_URLS)
        ]
        
        with patch("pathlib.Path.exists", side_effect=[True, False]):
            with patch("builtins.open", mock_open(read_data="/* cytoscape */")):
                logger = MagicMock()
                viz_dir = Path("/viz")
                
                result = load_vendor_content(viz_dir, file_list, logger, test_mode=False)
                
                assert "/* vendor/cytoscape.min.js */" in result
                assert "custom-plugin" not in result
                logger.warning.assert_any_call(f"Vendor file not found: {viz_dir}/vendor/custom-plugin.js")


@pytest.mark.viz
class TestGenerateTags:
    """Tests for generate_script_tags and generate_style_tags functions."""

    def test_generate_script_tags_cdn(self):
        """Test script tag generation for CDN mode."""
        file_list = ["vendor/cytoscape.min.js", "vendor/custom.js"]
        result = generate_script_tags(file_list, embed=False)

        assert 'src="https://unpkg.com/cytoscape' in result
        assert 'src="vendor/custom.js"' in result

    def test_generate_script_tags_embed(self):
        """Test script tag generation for embed mode."""
        file_list = ["vendor/cytoscape.min.js"]
        result = generate_script_tags(file_list, embed=True)

        assert result == ""  # Content embedded directly

    def test_generate_style_tags_cdn(self):
        """Test style tag generation for CDN mode."""
        file_list = ["vendor/cytoscape.js-navigator.css", "vendor/custom.css"]
        result = generate_style_tags(file_list, embed=False)

        assert 'href="https://unpkg.com/cytoscape.js-navigator' in result
        assert 'href="vendor/custom.css"' in result

    def test_generate_style_tags_embed(self):
        """Test style tag generation for embed mode."""
        file_list = ["vendor/styles.css"]
        result = generate_style_tags(file_list, embed=True)

        assert result == ""  # Content embedded directly


@pytest.mark.viz
class TestGenerateHtml:
    """Tests for generate_html function."""

    def test_generate_html_success(self):
        """Test successful HTML generation."""
        graph_data = {"nodes": [{"id": "1"}], "edges": []}
        concepts_data = {"concepts": []}
        config = {
            "graph2html": {
                "minify_json": True, 
                "embed_libraries": True,
                "vendor_js": [],
                "vendor_css": []
            },
            "visualization": {"initial_layout": "cose"},
            "colors": {"theme": "modern"},
            "ui": {"sidebar_width": 320},
            "node_shapes": {"chunk_shape": "ellipse"},
        }

        mock_template = MagicMock()
        mock_template.render.return_value = "<html>test</html>"

        with patch("jinja2.Environment.get_template", return_value=mock_template):
            with patch("builtins.open", mock_open(read_data="/* styles */")):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("viz.graph2html.load_vendor_content", return_value=""):
                        logger = MagicMock()
                        viz_dir = Path("/viz")

                        result = generate_html(graph_data, concepts_data, config, viz_dir, logger)

                        assert result == "<html>test</html>"
                        mock_template.render.assert_called_once()
                        
                        # Check context passed to template
                        call_kwargs = mock_template.render.call_args[1]
                        assert "graph_data_json" in call_kwargs
                        assert "concepts_data_json" in call_kwargs
                        assert "viz_config" in call_kwargs
                        assert "colors_config" in call_kwargs
                        assert "ui_config" in call_kwargs
                        assert "graph_stats" in call_kwargs
                        assert call_kwargs["graph_stats"]["nodes"] == 1
                        assert call_kwargs["graph_stats"]["edges"] == 0

    def test_generate_html_with_cose_bilkent_registration(self):
        """Test that cose-bilkent registration script is added."""
        graph_data = {"nodes": [], "edges": []}
        concepts_data = {"concepts": []}
        config = {
            "graph2html": {
                "minify_json": False,
                "embed_libraries": True,  # Embed mode
                "vendor_js": ["vendor/cytoscape.min.js", "vendor/cytoscape-cose-bilkent.js"],
                "vendor_css": []
            },
            "visualization": {},
            "colors": {},
            "ui": {},
            "node_shapes": {},
        }

        mock_template = MagicMock()
        mock_template.render.return_value = "<html>test</html>"
        
        with patch("jinja2.Environment.get_template", return_value=mock_template):
            with patch("builtins.open", mock_open(read_data="")):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("viz.graph2html.load_vendor_content", return_value="/* vendor libs */"):
                        logger = MagicMock()
                        viz_dir = Path("/viz")
                        
                        result = generate_html(graph_data, concepts_data, config, viz_dir, logger)
                        
                        # Check that vendor_js_content includes registration script
                        call_kwargs = mock_template.render.call_args[1]
                        vendor_js = call_kwargs["vendor_js_content"]
                        assert "cytoscape.use(cytoscapeCoseBilkent)" in vendor_js
                        assert "cose-bilkent layout registered successfully" in vendor_js

    def test_generate_html_cdn_mode_registration(self):
        """Test cose-bilkent registration in CDN mode."""
        graph_data = {"nodes": [], "edges": []}
        concepts_data = {"concepts": []}
        config = {
            "graph2html": {
                "minify_json": False,
                "embed_libraries": False,  # CDN mode
                "vendor_js": ["vendor/cytoscape.min.js", "vendor/cytoscape-cose-bilkent.js"],
                "vendor_css": []
            },
            "visualization": {},
            "colors": {},
            "ui": {},
            "node_shapes": {},
        }

        mock_template = MagicMock()
        mock_template.render.return_value = "<html>test</html>"
        
        with patch("jinja2.Environment.get_template", return_value=mock_template):
            with patch("builtins.open", mock_open(read_data="")):
                with patch("pathlib.Path.exists", return_value=True):
                    logger = MagicMock()
                    viz_dir = Path("/viz")
                    
                    result = generate_html(graph_data, concepts_data, config, viz_dir, logger)
                    
                    # Check that script_tags includes registration
                    call_kwargs = mock_template.render.call_args[1]
                    script_tags = call_kwargs["script_tags"]
                    assert "cytoscape.use(cytoscapeCoseBilkent)" in script_tags
                    assert "<script>" in script_tags

    def test_generate_html_template_error(self):
        """Test handling of template errors."""
        graph_data = {"nodes": [], "edges": []}
        concepts_data = {"concepts": []}
        config = {"graph2html": {"vendor_js": [], "vendor_css": []}}

        with patch("jinja2.Environment.get_template", side_effect=Exception("Template error")):
            logger = MagicMock()
            viz_dir = Path("/viz")

            with pytest.raises(SystemExit) as exc_info:
                generate_html(graph_data, concepts_data, config, viz_dir, logger)

            assert exc_info.value.code == 5  # EXIT_IO_ERROR
            logger.error.assert_called()

    def test_generate_html_graph_core_loading(self):
        """Test that graph_core.js is loaded and included."""
        graph_data = {"nodes": [], "edges": []}
        concepts_data = {"concepts": []}
        config = {
            "graph2html": {
                "embed_libraries": True,
                "vendor_js": [],
                "vendor_css": []
            },
            "visualization": {},
            "colors": {},
            "ui": {},
            "node_shapes": {},
        }

        mock_template = MagicMock()
        mock_template.render.return_value = "<html>test</html>"
        mock_graph_core = "// Graph core module code"
        
        # Use mock_open with specific content for graph_core.js
        with patch("jinja2.Environment.get_template", return_value=mock_template):
            with patch("builtins.open", mock_open(read_data=mock_graph_core)):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("viz.graph2html.load_vendor_content", return_value=""):
                        logger = MagicMock()
                        viz_dir = Path("/viz")
                        
                        result = generate_html(graph_data, concepts_data, config, viz_dir, logger)
                        
                        # Verify graph_core.js was loaded - check the actual call that was made
                        info_calls = [str(call) for call in logger.info.call_args_list]
                        assert any("Loaded graph_core.js" in str(call) for call in info_calls)
                        
                        # Check graph_core is passed separately to template
                        call_kwargs = mock_template.render.call_args[1]
                        # graph_core.js is now loaded separately and passed as graph_core_content
                        assert "graph_core_content" in call_kwargs
                        graph_core_content = call_kwargs["graph_core_content"]
                        # And the actual content should be there
                        assert mock_graph_core in graph_core_content


@pytest.mark.viz
class TestMinifyHtmlContent:
    """Tests for minify_html_content function."""

    def test_minify_html_enabled(self):
        """Test HTML minification when enabled."""
        html = """
        <html>
            <head>
                <title>Test</title>
            </head>
            <body>
                <div>Content</div>
            </body>
        </html>
        """
        config = {"graph2html": {"minify_html": True}}
        logger = MagicMock()

        result = minify_html_content(html, config, logger)

        # Check that result is minified
        assert len(result) < len(html)
        assert "\n        " not in result  # Indentation removed
        logger.info.assert_called()

    def test_minify_html_disabled(self):
        """Test HTML not minified when disabled."""
        html = "<html><body>Test</body></html>"
        config = {"graph2html": {"minify_html": False}}
        logger = MagicMock()

        result = minify_html_content(html, config, logger)

        assert result == html
        logger.info.assert_not_called()

    def test_minify_html_error_handling(self):
        """Test graceful handling of minification errors."""
        html = "<html><body>Test</body></html>"
        config = {"graph2html": {"minify_html": True}}
        logger = MagicMock()

        with patch("minify_html.minify", side_effect=Exception("Minify error")):
            result = minify_html_content(html, config, logger)

            assert result == html  # Returns original on error
            logger.warning.assert_called()


@pytest.mark.viz
class TestSaveHtml:
    """Tests for save_html function."""

    def test_save_html_success(self):
        """Test successful HTML saving."""
        html = "<html><body>Test</body></html>"
        output_path = Path("/test/output.html")

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1024
                    logger = MagicMock()

                    save_html(html, output_path, logger)

                    mock_file.assert_called_once_with(output_path, "w", encoding="utf-8")
                    mock_file().write.assert_called_once_with(html)
                    logger.info.assert_called()

    def test_save_html_io_error(self):
        """Test handling of I/O errors."""
        html = "<html></html>"
        output_path = Path("/test/output.html")

        with patch("builtins.open", side_effect=OSError("Write error")):
            logger = MagicMock()

            with pytest.raises(SystemExit) as exc_info:
                save_html(html, output_path, logger)

            assert exc_info.value.code == 5  # EXIT_IO_ERROR
            logger.error.assert_called()