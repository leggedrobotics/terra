import unittest

from src.actions import TrackedAction
from src.actions import TrackedActionType


class TestActions(unittest.TestCase):
    def test_tracked_action(self):
        action = TrackedAction.new(TrackedActionType.DO)

        self.assertTrue(action.action.item() == TrackedActionType.DO)


if __name__ == "__main__":
    unittest.main()
