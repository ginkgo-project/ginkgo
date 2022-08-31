import unittest
import pygko
import numpy as np


class ExecutorTest(unittest.TestCase):
    def test_can_create_executor(self):
        ref = pygko.ReferenceExecutor()
        self.assertIsNotNone(ref)


class ArrayTest(unittest.TestCase):
    array_length = 10
    ref = pygko.ReferenceExecutor()

    def test_can_create_arrays(self):
        arr1 = pygko.array(self.ref, self.array_length)
        arr2 = pygko.array(self.ref)

    def test_basic_array_functions(self):
        ref = pygko.ReferenceExecutor()
        arr = pygko.array(self.ref, self.array_length)
        arr.fill(1)
        self.assertIs(arr.get_num_elems(), self.array_length)
        self.assertIs(pygko.reduce_add(arr), self.array_length * 1)

    def test_array_numpy_interop(self):
        np_arr = np.array(range(10))

        gko_arr = pygko.array(self.ref, self.array_length, np_arr)

        np_arr_from_gko = np.array(gko_arr, copy=False)


class DenseTest(unittest.TestCase):
    values = [1, 2, -1, 3, 4, -1, 5, 6, -1]
    array_length = 10
    ref = pygko.ReferenceExecutor()

    def test_can_create_dense_linop(self):
        dense = pygko.array(self.ref, (self.array_length))

    def test_can_create_dense_from_array(self):
        arr = pygko.array(self.ref, 9, np.array([self.values]))
        dense = pygko.matrix.Dense(self.ref, (3, 2), arr, 3)

        self.assertIs(dense[2], 5)
        self.assertIs(dense[2, 1], 6)

    def test_can_create_dense_from_list(self):
        dense = pygko.matrix.Dense(self.ref, (3, 2), self.values, 3)

        self.assertIs(dense.at(2, 1), 6)

    def test_can_create_dense_from_nparray(self):
        dense = pygko.Dense(self.ref, (3, 2), np.array(self.values), 3)

        self.assertIs(dense.at(2, 1), 6)

    def test_dense_support_basic_functionality(self):
        np_array = np.array(self.values)
        dense = pygko.matrix.Dense(self.ref, (9, 1), np_array, 1)

        result = pygko.matrix.Dense(self.ref, (1, 1))

        dense.compute_norm1(result)
        self.assertIs(result[0, 0], np.linalg.norm(np_array), 1)

        dense.compute_norm2(result)
        self.assertIs(result[0, 0], np.linalg.norm(np_array))

        # TODO for now just test if callable add checking of results
        dense.scale(result)
        dense.inv_scale(result)
        dense.add_scale(result)
        dense.sub_scale(result)
        dense.compute_dot(dense, result)


class CsrTest(unittest.TestCase):
    values = [1, 2, 3, 4]
    cols = [0, 1, 1, 0]
    rows = [0, 2, 3, 4]
    array_length = 4
    ref = pygko.ReferenceExecutor()

    def test_can_create_from_arrays(self):

        mtx = pygko.csr(
            self.ref,
            (3, 2),
            pygko.array(self.ref, 4, self.values),
            pygko.array(self.ref, 4, self.cols),
            pygko.array(self.ref, 4, self.rows),
        )


if __name__ == "__main__":
    unittest.main()
