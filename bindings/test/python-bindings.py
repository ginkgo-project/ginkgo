import unittest
import pygko


class MainTest(unittest.TestCase):
    def test_can_create_executor(self):
        ref = pygko.ReferenceExecutor()
        self.assertIsNotNone(ref)

    def test_can_create_arrays(self):
        ref = pygko.ReferenceExecutor()
        ar = pygko.array(ref, 10)


if __name__ == "__main__":
    unittest.main()
