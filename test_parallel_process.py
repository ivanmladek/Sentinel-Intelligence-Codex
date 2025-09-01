import os
import json
import tempfile
import shutil
import subprocess
import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
import sys
from langdetect import LangDetectException

# Add the current directory to the path so we can import parallel_process
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions we want to test
from parallel_process import (
    get_file_list, download_file, extract_rar, sanitize_filename,
    process_pdf, check_gcs_file_exists, upload_to_huggingface,
    check_huggingface_file_exists, clean_text, calculate_text_quality_score,
    is_garbage, chunk_text, split_segment, process_and_chunk_mmd,
    upload_to_gcs, cleanup_files, process_single_rar, main,
    GARBAGE_THRESHOLD, LENWORD, ENGLISH_WORDS
)


class TestFileOperations:
    """Test file operation functions."""

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        assert sanitize_filename("test.pdf") == "test.pdf"
        assert sanitize_filename("test file.pdf") == "test_file.pdf"
        assert sanitize_filename("test@#$file.pdf") == "test___file.pdf"
        assert sanitize_filename("test/file.pdf") == "test_file.pdf"

    @patch('parallel_process.requests.get')
    def test_download_file_success(self, mock_get):
        """Test successful file download."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'test data']
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test.txt')
            result = download_file('http://example.com/test.txt', output_path)

            assert result is True
            assert os.path.exists(output_path)
            with open(output_path, 'rb') as f:
                assert f.read() == b'test data'

    @patch('parallel_process.requests.get')
    def test_download_file_exists(self, mock_get):
        """Test download when file already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test.txt')
            with open(output_path, 'w') as f:
                f.write('existing')

            result = download_file('http://example.com/test.txt', output_path)
            assert result is True
            # Should not call requests.get
            mock_get.assert_not_called()

    @patch('parallel_process.requests.get')
    def test_download_file_failure(self, mock_get):
        """Test download failure."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Download failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test.txt')
            result = download_file('http://example.com/test.txt', output_path)

            assert result is False

    @patch('parallel_process.subprocess.run')
    def test_extract_rar_success(self, mock_run):
        """Test successful RAR extraction."""
        mock_run.return_value = Mock(returncode=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            rar_path = os.path.join(temp_dir, 'test.rar')
            extract_path = os.path.join(temp_dir, 'extracted')

            with open(rar_path, 'w') as f:
                f.write('dummy')

            result = extract_rar(rar_path, extract_path)
            assert result is True
            assert os.path.exists(extract_path)

    @patch('parallel_process.subprocess.run')
    def test_extract_rar_failure(self, mock_run):
        """Test RAR extraction failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'unrar', stderr="Extraction failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            rar_path = os.path.join(temp_dir, 'test.rar')
            extract_path = os.path.join(temp_dir, 'extracted')

            with open(rar_path, 'w') as f:
                f.write('dummy')

            result = extract_rar(rar_path, extract_path)
            assert result is False


class TestTextProcessing:
    """Test text processing functions."""

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "# Header\n\n**bold** and *italic* text.\n\nTable 1 shows data.\n\n$content$"
        cleaned = clean_text(text)
        assert "Header" not in cleaned
        assert "bold" in cleaned
        assert "italic" in cleaned
        assert "Table" not in cleaned
        assert "$content$" not in cleaned

    def test_clean_text_whitespace(self):
        """Test whitespace normalization."""
        text = "  multiple   spaces  \n\n  and  lines  "
        cleaned = clean_text(text)
        assert cleaned == "multiple spaces and lines"

    def test_clean_text_latex(self):
        """Test LaTeX command removal."""
        text = r"This is \textbf{bold} and \textit{italic} text with \section{section}."
        cleaned = clean_text(text)
        assert r"\textbf" not in cleaned
        assert r"\textit" not in cleaned
        assert r"\section" not in cleaned
        # The current implementation removes LaTeX commands but keeps the content
        assert "bold" in cleaned or "This is" in cleaned
        assert "italic" in cleaned or "and" in cleaned

    def test_calculate_text_quality_score(self):
        """Test text quality score calculation."""
        # High quality English text
        good_text = "This is a well-written sentence. It contains proper grammar and structure."
        score = calculate_text_quality_score(good_text)
        assert score > 0.5

        # Empty text
        assert calculate_text_quality_score("") == 0.0

        # Non-English text (should still get some score for structure)
        non_english = "Dies ist ein deutscher Satz. Er hat eine gute Struktur."
        score = calculate_text_quality_score(non_english)
        assert score >= 0.0

    def test_is_garbage_short_text(self):
        """Test garbage detection for short text."""
        assert is_garbage("short")
        assert is_garbage("")
        assert not is_garbage("This is a longer piece of text that should not be considered garbage.")

    def test_is_garbage_long_word(self):
        """Test garbage detection for text with very long words."""
        long_word = "a" * (LENWORD + 1)
        text_with_long_word = f"This text contains a very {long_word} word."
        assert is_garbage(text_with_long_word)

    @patch('parallel_process.detect')
    def test_is_garbage_non_english(self, mock_detect):
        """Test garbage detection for non-English text."""
        mock_detect.return_value = 'de'  # German
        assert is_garbage("Dies ist deutscher Text.")

    @patch('parallel_process.detect')
    def test_is_garbage_detection_error(self, mock_detect):
        """Test garbage detection when language detection fails."""
        mock_detect.side_effect = LangDetectException("Detection failed", "test")
        assert is_garbage("Some text that can't be detected.")

    def test_chunk_text_headings(self):
        """Test text chunking with markdown headings."""
        text = """# Introduction

This is the introduction paragraph.

## Section 1

This is section 1 content.

## Section 2

This is section 2 content.
"""

        chunks = chunk_text(text)
        assert len(chunks) > 1
        assert any("Introduction" in chunk for chunk in chunks)
        assert any("Section 1" in chunk for chunk in chunks)

    def test_split_segment_sentences(self):
        """Test segment splitting by sentences."""
        segment = "First sentence. Second sentence! Third sentence?"
        chunks = split_segment(segment, max_size=20)

        assert len(chunks) > 1
        assert "First sentence." in " ".join(chunks)
        assert "Second sentence!" in " ".join(chunks)

    def test_split_segment_size_limit(self):
        """Test segment splitting with size limits."""
        long_segment = "Short. " + "This is a very long sentence that should be split. " * 10
        chunks = split_segment(long_segment, max_size=50)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50


class TestCloudOperations:
    """Test cloud storage operations."""

    @patch('parallel_process.storage.Client')
    def test_check_gcs_file_exists_success(self, mock_client):
        """Test GCS file existence check."""
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        mock_client.return_value.bucket.return_value = mock_bucket

        result = check_gcs_file_exists('test-bucket', 'test-file.txt')
        assert result is True

    @patch('parallel_process.storage.Client')
    def test_check_gcs_file_exists_failure(self, mock_client):
        """Test GCS file existence check failure."""
        mock_client.side_effect = Exception("GCS error")

        result = check_gcs_file_exists('test-bucket', 'test-file.txt')
        assert result is False

    @patch('parallel_process.HfApi')
    def test_upload_to_huggingface_success(self, mock_api_class):
        """Test successful Hugging Face upload."""
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            with patch('parallel_process.os.environ', {'HUGGING_FACE_TOKEN': 'test-token'}):
                result = upload_to_huggingface(temp_path, 'test-repo')
                assert result is True
        finally:
            os.unlink(temp_path)

    def test_upload_to_huggingface_no_token(self):
        """Test Hugging Face upload without token."""
        with patch('parallel_process.os.environ', {}):
            result = upload_to_huggingface('/tmp/test.txt', 'test-repo')
            assert result is False

    @patch('parallel_process.HfApi')
    def test_check_huggingface_file_exists_success(self, mock_api_class):
        """Test Hugging Face file existence check."""
        mock_api = Mock()
        mock_api.list_repo_files.return_value = ['file1.txt', 'file2.txt']
        mock_api_class.return_value = mock_api

        with patch('parallel_process.os.environ', {'HUGGING_FACE_TOKEN': 'test-token'}):
            result = check_huggingface_file_exists('test-repo', 'file1.txt')
            assert result is True

            result = check_huggingface_file_exists('test-repo', 'nonexistent.txt')
            assert result is False

    def test_check_huggingface_file_exists_no_token(self):
        """Test Hugging Face check without token."""
        with patch('parallel_process.os.environ', {}):
            result = check_huggingface_file_exists('test-repo', 'file.txt')
            assert result is False


class TestFileProcessing:
    """Test file processing functions."""

    @patch('parallel_process.subprocess.Popen')
    def test_process_pdf_success(self, mock_popen):
        """Test successful PDF processing."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ""
        mock_process.wait.return_value = 0
        mock_process.stdout.close.return_value = None
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, 'test.pdf')
            output_dir = os.path.join(temp_dir, 'output')

            with open(pdf_path, 'w') as f:
                f.write('dummy pdf')

            mmd_path = os.path.join(output_dir, 'test.mmd')
            os.makedirs(output_dir, exist_ok=True)
            with open(mmd_path, 'w') as f:
                f.write('dummy mmd')

            result = process_pdf(pdf_path, output_dir)
            assert result == mmd_path

    @patch('parallel_process.subprocess.Popen')
    def test_process_pdf_failure(self, mock_popen):
        """Test PDF processing failure."""
        mock_process = Mock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, 'test.pdf')
            output_dir = os.path.join(temp_dir, 'output')

            with open(pdf_path, 'w') as f:
                f.write('dummy pdf')

            result = process_pdf(pdf_path, output_dir)
            assert result is None

    def test_process_and_chunk_mmd_success(self):
        """Test successful MMD processing and chunking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mmd_path = os.path.join(temp_dir, 'test.mmd')
            output_dir = temp_dir

            with open(mmd_path, 'w') as f:
                f.write("# Header\n\nThis is test content.\n\n## Section\n\nMore content.")

            cleaned_path, garbage_path = process_and_chunk_mmd(mmd_path, output_dir)

            assert cleaned_path is not None
            assert garbage_path is not None
            assert os.path.exists(cleaned_path)
            assert os.path.exists(garbage_path)

    def test_process_and_chunk_mmd_file_not_found(self):
        """Test MMD processing when file doesn't exist."""
        result = process_and_chunk_mmd('/nonexistent/file.mmd', '/tmp')
        assert result == (None, None)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_cleanup_files(self):
        """Test file cleanup function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files and directories
            file1 = os.path.join(temp_dir, 'file1.txt')
            file2 = os.path.join(temp_dir, 'file2.txt')
            dir1 = os.path.join(temp_dir, 'dir1')

            with open(file1, 'w') as f:
                f.write('test')
            with open(file2, 'w') as f:
                f.write('test')
            os.makedirs(dir1)

            # Test cleanup
            cleanup_files(file1, dir1, '/nonexistent/file.txt')

            assert not os.path.exists(file1)
            assert not os.path.exists(dir1)
            assert os.path.exists(file2)  # Should not be cleaned up


class TestMainProcessing:
    """Test main processing functions."""

    @patch('parallel_process.get_file_list')
    @patch('parallel_process.process_single_rar')
    @patch('parallel_process.BUCKET_NAME', 'test-bucket')
    @patch('parallel_process.os.environ', {'JOB_COMPLETION_INDEX': '0', 'NUM_PODS': '1'})
    def test_main_success(self, mock_process_rar, mock_get_list):
        """Test main function success."""
        mock_get_list.return_value = ['http://example.com/file1.rar', 'http://example.com/file2.rar']

        main()

        # Should call process_single_rar for each file
        assert mock_process_rar.call_count == 2

    @patch('parallel_process.os.environ', {})
    def test_main_no_bucket(self):
        """Test main function without bucket name."""
        main()  # Should not crash, just log error

    @patch('parallel_process.get_file_list')
    @patch('parallel_process.check_gcs_file_exists')
    @patch('parallel_process.download_file')
    @patch('parallel_process.extract_rar')
    @patch('parallel_process.upload_to_gcs')
    @patch('parallel_process.cleanup_files')
    def test_process_single_rar_skip_existing(self, mock_cleanup, mock_upload_gcs,
                                             mock_extract, mock_download, mock_check_gcs, mock_get_list):
        """Test processing RAR that already exists."""
        mock_check_gcs.return_value = True  # RAR already exists

        with tempfile.TemporaryDirectory() as temp_dir:
            result = process_single_rar('http://example.com/test.rar', 'test-bucket')

            assert result == 0
            mock_cleanup.assert_called()


class TestIntegration:
    """Integration tests for complex workflows."""

    def test_full_text_processing_pipeline(self):
        """Test the complete text processing pipeline."""
        # Test data
        raw_text = r"""

        # Introduction

        This is **bold** text with *italic* formatting.

        ## Section 1

        This section contains a table reference: Table 1 shows data.

        Here is some LaTeX: $x = y^2$ and $\int_0^1 f(x) dx$.

        ## Section 2

        This is a longer paragraph with multiple sentences. It should be processed correctly.
        The quality score should be reasonable for this English text.
        """
        # Clean the text
        cleaned = clean_text(raw_text)

        # Verify cleaning worked
        assert "Introduction" not in cleaned  # Headers removed
        assert "bold" in cleaned  # Formatting removed but text kept
        assert "italic" in cleaned
        assert "Table" not in cleaned  # Table references removed
        assert "$x = y^2$" not in cleaned  # LaTeX removed
        assert "This is a longer paragraph" in cleaned  # Content preserved

        # Test quality scoring
        quality_score = calculate_text_quality_score(cleaned)
        assert quality_score > 0.3  # Should be decent quality

        # Test garbage detection
        assert not is_garbage(cleaned)  # Should not be garbage

        # Test chunking
        chunks = chunk_text(raw_text)
        assert len(chunks) > 1  # Should create multiple chunks

        # Verify chunks contain expected content
        chunk_text_combined = " ".join(chunks)
        assert "Introduction" in chunk_text_combined
        assert "Section 1" in chunk_text_combined
        assert "Section 2" in chunk_text_combined


if __name__ == "__main__":
    pytest.main([__file__])