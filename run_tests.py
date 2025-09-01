#!/usr/bin/env python3
"""
Test runner for parallel_process.py

This script runs the comprehensive test suite for the PDF processing pipeline.
"""

import sys
import subprocess
import os

def install_test_dependencies():
    """Install test dependencies if not already installed."""
    try:
        import pytest
        import pytest_mock
        print("✅ Test dependencies already installed")
    except ImportError:
        print("📦 Installing test dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "test_requirements.txt"
        ])
        print("✅ Test dependencies installed")

def run_tests():
    """Run the test suite."""
    print("🧪 Running tests for parallel_process.py...")
    print("=" * 50)

    # Run pytest with verbose output
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "test_parallel_process.py",
        "-v",
        "--tb=short",
        "--color=yes"
    ], capture_output=False)

    return result.returncode

def run_coverage():
    """Run tests with coverage if pytest-cov is available."""
    try:
        import pytest_cov
        print("\n📊 Running tests with coverage...")
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "test_parallel_process.py",
            "--cov=parallel_process",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "-v"
        ], capture_output=False)
        return result.returncode
    except ImportError:
        print("\n⚠️  pytest-cov not installed. Install with: pip install pytest-cov")
        return 0

def main():
    """Main test runner function."""
    print("🚀 PDF Processing Pipeline Test Runner")
    print("=" * 50)

    # Check if test file exists
    if not os.path.exists("test_parallel_process.py"):
        print("❌ test_parallel_process.py not found!")
        sys.exit(1)

    if not os.path.exists("parallel_process.py"):
        print("❌ parallel_process.py not found!")
        sys.exit(1)

    # Install dependencies
    install_test_dependencies()

    # Run tests
    exit_code = run_tests()

    # Run coverage if tests passed
    if exit_code == 0:
        run_coverage()

    # Summary
    print("\n" + "=" * 50)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Exit code: {exit_code}")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()