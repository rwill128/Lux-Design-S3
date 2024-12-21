#!/usr/bin/env python3
"""Test runner script for reward tile verification tests.

This script runs the reward tile verification tests using pytest
and ensures proper exit code propagation for CI/CD integration.
"""
import pytest
import sys

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "test_reward_tile_verification.py"]))
