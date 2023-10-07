# Docs: https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option

import pytest
from itertools import product

def pytest_addoption(parser):
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="Runs slow tests."
    )
    parser.addoption(
        "--rerun_files", action="store_true", default=False, help="Runs tests that overwrite large files."
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "overwrites: mark test as slow and overwriting files")

def pytest_collection_modifyitems(config, items):
    skips = {}
    if not config.getoption("--run_slow"):
        skips.update({"slow": pytest.mark.skip(reason="Run with --run_slow")})
    if not config.getoption("--rerun_files"):
        skips.update({"overwrites": pytest.mark.skip(reason="Run with --rerun_files")})
    for item, reason in product(items, skips.keys()):
        if reason in item.keywords:
            item.add_marker(skips[reason])

# # each test runs on cwd to its temp dir
# @pytest.fixture(autouse=True)
# def go_to_tmpdir(request):
#     # Get the fixture dynamically by its name.
#     tmpdir = request.getfixturevalue("tmpdir")
#     # ensure local test created packages can be imported
#     sys.path.insert(0, str(tmpdir))
#     # Chdir only for the duration of the test.
#     with tmpdir.as_cwd():
#         yield