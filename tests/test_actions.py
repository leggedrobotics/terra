import unittest
import jax.numpy as jnp
from src.actions import TrackedAction, TrackedActionType


class TestActions(unittest.TestCase):

    def test_tracked_action(self):
        action = TrackedAction.new(TrackedActionType.DO)

        self.assertTrue(action.action.item() == TrackedActionType.DO)

        # print(action)

if __name__ == "__main__":
    unittest.main()
