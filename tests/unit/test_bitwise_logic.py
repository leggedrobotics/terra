import unittest


class TestBitwiseLogic(unittest.TestCase):
    @staticmethod
    def _bitwise_in(a: tuple[int, int, int], b: int):
        return (b == a[0]) | (b == a[1]) | (b == a[2])

    def test_bitwise_logic_1(self):
        a = (0, 2, 3)
        b = 2
        c = 5

        logic_b = b in a
        logic_c = c in a

        bitwise_b = self._bitwise_in(a, b)
        bitwise_c = self._bitwise_in(a, c)

        self.assertEqual(logic_b, bitwise_b)
        self.assertEqual(logic_c, bitwise_c)


if __name__ == "__main__":
    unittest.main()
