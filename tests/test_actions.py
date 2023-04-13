import unittest
import jax.numpy as jnp
from src.actions import ActionBatch


class TestActions(unittest.TestCase):

    def test_action_batch(self):
        capacity = 5
        action_batch = ActionBatch.empty(capacity)

        self.assertTrue(action_batch.data.action.shape[0] == capacity)
        self.assertTrue(action_batch.data.action.sum() == -capacity)


if __name__ == "__main__":
    unittest.main()
