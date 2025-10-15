# graph.py
import numpy as np

class Graph:
    """ The Graph to model the skeletons of PKU-MMD dataset
    Format: (channel, vertex, vertex)
    """
    def __init__(self, labeling_mode='spatial'):
        self.num_node = 25
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                               (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                               (11, 10), (12, 11), (13, 1), (14, 13),
                               (15, 14), (16, 15), (17, 1), (18, 17),
                               (19, 18), (20, 19), (22, 23), (23, 8),
                               (24, 25), (25, 12)]
        self.inward = [(i - 1, j - 1) for (i, j) in self.inward_ori_index]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = self.get_spatial_graph()
        else:
            raise ValueError()
        return A

    def get_spatial_graph(self):
        I = np.eye(self.num_node)
        A_outward = np.zeros((self.num_node, self.num_node))
        A_inward = np.zeros((self.num_node, self.num_node))
        for i, j in self.self_link:
            I[i, j] = 1
        for i, j in self.inward:
            A_inward[i, j] = 1
        for i, j in self.outward:
            A_outward[i, j] = 1
        
        A = A_inward + A_outward + I
        # Normalize the adjacency matrix
        D = np.sum(A, 0)
        D[D <= 10e-4] = 1
        D = np.diag(D)
        A = np.dot(A, np.linalg.inv(D))
        
        return np.stack([I, A_inward, A_outward])