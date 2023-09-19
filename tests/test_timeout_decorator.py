from idr_design.timeout_decorator import timeout
from time import sleep
import pytest

@timeout(1)
def sleep_and_succeed_no_decorator(seconds: float) -> int:
    sleep(seconds)
    return 0

@timeout(1)
def sleep_and_fail_no_decorator(seconds: float) -> int:
    sleep(seconds)
    raise ValueError()

@timeout(1,"Custom error message!")
def timeout_with_decorator(seconds: int) -> int:
    sleep(seconds)
    return 0

@pytest.mark.slow
class TestDecorator:
    def test_timeout_before_succeed(self):
        with pytest.raises(RuntimeError):
            sleep_and_fail_no_decorator(1.1)
    def test_timeout_before_fail(self):
        with pytest.raises(RuntimeError):
            sleep_and_succeed_no_decorator(1.1)
    def test_succeed(self):
        assert sleep_and_succeed_no_decorator(0.1) == 0
    def test_fail(self):
        with pytest.raises(ValueError):
            sleep_and_fail_no_decorator(0.1)
    def test_custom_decorator(self):
        with pytest.raises(RuntimeError) as e:
            timeout_with_decorator(1.1)
        assert e.value.args[0] == "Custom error message!"

