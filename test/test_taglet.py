import unittest
from mrftools import *
import numpy as np


class TestMarkovNet(unittest.TestCase):

    def create_chain_model(self):
        """Test basic functionality of BeliefPropagator."""
        mn = MarkovNet()

        np.random.seed(1)

        k = [4, 3, 6, 2, 5]

        mn.set_unary_factor(0, np.random.randn(k[0]))
        mn.set_unary_factor(1, np.random.randn(k[1]))
        mn.set_unary_factor(2, np.random.randn(k[2]))
        mn.set_unary_factor(3, np.random.randn(k[3]))

        factor4 = np.random.randn(k[4])
        factor4[2] = -float('inf')

        mn.set_unary_factor(4, factor4)

        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((3, 2), np.random.randn(k[3], k[2]))
        mn.set_edge_factor((1, 4), np.random.randn(k[1], k[4]))

        return mn

    def test_structure(self):
        mn = MarkovNet()

        mn.set_unary_factor(0, np.random.randn(4))
        mn.set_unary_factor(1, np.random.randn(3))
        mn.set_unary_factor(2, np.random.randn(5))

        mn.set_edge_factor((0, 1), np.random.randn(4, 3))
        mn.set_edge_factor((1, 2), np.random.randn(3, 5))

        print("Neighbors of 0: " + repr(mn.get_neighbors(0)))
        print("Neighbors of 1: " + repr(mn.get_neighbors(1)))
        print("Neighbors of 2: " + repr(mn.get_neighbors(2)))

        assert mn.get_neighbors(0) == set([1]), "Neighbors are wrong"
        assert mn.get_neighbors(1) == set([0, 2]), "Neighbors are wrong"
        assert mn.get_neighbors(2) == set([1]), "Neighbors are wrong"

    def test_matrix_shapes(self):
        mn = self.create_chain_model()

        k = [4, 3, 6, 2, 5]

        max_states = max(k)

        assert mn.matrix_mode == False, "Matrix mode flag was set prematurely"

        mn.create_matrices()

        assert mn.matrix_mode, "Matrix mode flag wasn't set correctly"

        assert mn.unary_mat.shape == (max_states, 5)
