# Docs: https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option

import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="Runs slow tests."
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "slow_but_run_anyway: ignore the slow marker")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_slow"):
        return
    skip_slow = pytest.mark.skip(reason="Test(s) is/are slow and shouldn't be run everytime. (Run with --run_slow)")
    for item in items:
        if "slow" in item.keywords and "slow_but_run_anyway" not in item.keywords:
            item.add_marker(skip_slow)

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