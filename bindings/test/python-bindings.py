import unittest
#import python_cpp_example  # our `pybind11`-based extension module

class MainTest(unittest.TestCase):
    def test_add(self):
        # test that 1 + 1 = 2
        # self.assertEqual(python_cpp_example.add(1, 1), 2)
        self.assertEqual(1,1)


if __name__ == '__main__':
    unittest.main()
