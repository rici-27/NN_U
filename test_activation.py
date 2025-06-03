import unittest
from layer import Softmax_Layer as Softmax
from layer import ACT_Layer_sigmoid as Sigmoid
from layer import ACT_Layer_ReLu as ReLU
from tensor import Tensor
import numpy as np

# Tests wurden für unseren Code angepasst

class TestSoftmax(unittest.TestCase):
    def setUp(self) -> None:
        self.softmax = Softmax(inShape=4)

    def test_forward(self) -> None:
        in_tensors = Tensor(elements=np.array([1, 2, 3, 4], dtype=np.float64))
        out_tensors = Tensor(elements=np.array([0, 0, 0, 0], dtype=np.float64))
        self.softmax.forward(inTensor= in_tensors, outTensor = out_tensors)
        print(out_tensors.elements)
        self.assertTrue(
            np.allclose(
                out_tensors.elements,
                np.array([0.0320586, 0.08714432, 0.23688282, 0.64391426]),
                rtol=1e-05,
                atol=1e-08,
            ),
            "forward softmax function does not calculate the correct outputs",
        )

    def test_backward(self) -> None:
        in_tensors = Tensor(elements=np.array([1, 2, 3, 4], dtype=np.float64))
        out_tensors = Tensor(elements=np.array([0, 0, 0, 0], dtype=np.float64))
        out_tensors.deltas = np.array([6, 7, 8, 9])

        sm_diag = np.array([[0.0320586, 0, 0, 0], [0, 0.08714432, 0, 0], [0, 0, 0.23688282, 0], [0, 0, 0, 0.64391426]])
        sm_mat = np.array([[0.001027753833960000, 0.002793725, 0.007594132, 0.020642989695636000],
                           [0.002793724897152000, 0.007594133, 0.020642992, 0.056113470326003200],
                           [0.007594131573252000, 0.020642992, 0.05611347, 0.152532225747013000],
                           [0.020642989695636000, 0.05611347, 0.152532226, 0.414625574231348000]])

        target_values = np.dot(out_tensors.deltas, (sm_diag - sm_mat))
        print(f"targets: {target_values}")
        self.softmax.forward(in_tensors, out_tensors)
        self.softmax.backward(outTensor=out_tensors, inTensor=in_tensors)
        print(f"calculation: {in_tensors.deltas}")
        self.assertTrue(
            np.allclose(
                in_tensors.deltas,
                target_values,
                rtol=1e-05,
                atol=1e-08,
            ),
            "backward softmax function does not calculate the correct outputs",
        )


class TestSigmoid(unittest.TestCase):
    def setUp(self) -> None:
        self.sigmoid = Sigmoid((5,))

    def test_forward(self) -> None:
        in_tensors = Tensor(elements=np.array([-2, 2, 0, 4, 5], dtype=np.float64))
        out_tensors = Tensor(elements=np.array([0, 0, 0, 0, 0], dtype=np.float64))
        expected_output = np.array([0.11920292, 0.88079708, 0.5, 0.98201379, 0.99330715])

        self.sigmoid.forward(in_tensors, out_tensors)
        self.assertTrue(
            np.allclose(
                out_tensors.elements,
                expected_output,
                rtol=1e-05,
                atol=1e-08,
            ),
            "forward sigmoid function does not calculate the correct outputs",
        )

    def test_backward(self) -> None:
        in_tensors = Tensor(elements=np.array([1, 2, 3, 4], dtype=np.float64))
        out_tensors = Tensor(elements=np.array([0, 0, 0, 0], dtype=np.float64))
        out_tensors.deltas = np.array([6, 7, 8, 9])
        expected_output = [1.1796716, 0.7349551, 0.36141328, 0.15896436]

        self.sigmoid.forward(in_tensors, out_tensors)
        self.sigmoid.backward(outTensor=out_tensors, inTensor =in_tensors)
        self.assertTrue(
            np.allclose(
                in_tensors.deltas,
                expected_output,
                rtol=1e-05,
                atol=1e-08,
            ),
            "backward sigmoid function does not calculate the correct outputs",
        )


class TestReLU(unittest.TestCase):
    def setUp(self) -> None:
        self.relu = ReLU(5)

    def test_forward(self) -> None:
        in_tensors = Tensor(elements=np.array([-2, 2, 0, 4, 5], dtype=np.float64))
        out_tensors = Tensor(elements=np.array([0, 0, 0, 0, 0], dtype=np.float64))
        expected_output = np.array([0, 2, 0, 4, 5])

        self.relu.forward(in_tensors, out_tensors)
        self.assertTrue(
            np.allclose(
                out_tensors.elements,
                expected_output,
                rtol=1e-05,
                atol=1e-08,
            ),
            "forward relu function does not calculate the correct outputs",
        )

    def test_backward(self) -> None:
        in_tensors = Tensor(elements=np.array([-1, 2, -3, 4], dtype=np.float64))
        out_tensors = Tensor(elements=np.array([0, 0, 0, 0], dtype=np.float64))
        out_tensors.deltas = np.array([-3, -7, 8, 9])
        expected_output = np.array([0, -7, 0, 9])

        self.relu.forward(in_tensors, out_tensors)
        self.relu.backward(outTensor=out_tensors, inTensor=in_tensors)
        self.assertTrue(
            np.allclose(
                in_tensors.deltas,
                expected_output,
                rtol=1e-05,
                atol=1e-08,
            ),
            "backward relu function does not calculate the correct outputs",
        )


if __name__ == "__main__":
    unittest.main()
