##  Main function of BGSR framework for brain graph super-resolution .
#  Details can be found in the original paper:Brain Graph Super-Resolution for Boosting Neurological
#  Disorder Diagnosis using Unsupervised Multi-Topology Residual Graph Manifold Learning.
#
#
#    ---------------------------------------------------------------------
#
#      This file contains the implementation of the key steps of our BGSR framework:
#      (1) Estimation of a connectional brain template (CBT)
#      (2) Proposed CBT-guided graph super-resolution :
#
#                        [pHR] = BGSR(train_data,train_Labels,HR_features,kn)
#
#                  Inputs:
#
#                           train_data: ((n-1) × m × m) tensor stacking the symmetric matrices of the training subjects (LR
#                           graphs)
#                                       n the total number of subjects
#                                       m the number of nodes
#
#                           train_Labels: ((n-1) × 1) vector of training labels (e.g., -1, 1)
#
#                           HR_features:   (n × (m × m)) matrix stacking the source HR brain graph.
#
#                           Kn: Number of most similar LR training subjects.
#
#
#                  Outputs:
#                          pHR: (1 × (m × m)) vector stacking the predicted features of the testing subject.
#
#
#
#      To evaluate our framework we used Leave-One-Out cross validation strategy.
#
#
#
# To test BGSR on random data, we defined the function 'simulateData_LR_HR' where the size of the dataset is chosen by the user.
#  ---------------------------------------------------------------------
#      Copyright 2019 Islem Mhiri, Sousse University.
#      Please cite the above paper if you use this code.
#      All rights reserved.
#      """
#
#   ------------------------------------------------------------------------------

import numpy as np
import snf
import SIMLR_PY.SIMLR
from atlas import atlas
import networkx as nx
import matplotlib.pyplot as plt

def BGSR(train_data,train_labels,HR_features,kn):

    def isDirected(adj):
        S = True
        for i in range(len(adj)):
            for j in range(len(adj[0])):
                if adj[j][i] == np.transpose(adj)[j][i]:
                    S = False
        return S

    def degrees(adj):

        indeg = np.sum(adj, axis=1)
        outdeg = np.sum(np.transpose(adj), axis=0)

        if isDirected(adj):
            deg = indeg + outdeg #total degree
        else: #undirected graph: indeg=outdeg
            deg = indeg + np.transpose(np.diag(adj)) #add self-loops twice, if any

        return deg

    #This is a reproduction of the closeness function of aeolianine since I couldn't find a compatible closeness function in python.
    def closeness(G, adj):

        c = np.zeros((len(adj),1))
        all_sum = np.zeros((len(adj[1]),1))
        spl = nx.all_pairs_dijkstra_path_length(G, weight="weight")
        spl_list = list(spl)
        for i in range(len(adj[1])):
            spl_dict = spl_list[i][1]
            c[i] = 1 / sum(spl_dict.values())
        return c

    sz1, sz2, sz3 = train_data.shape
    # (1) Estimation of a connectional brain template (CBT)
    CBT = atlas(train_data, train_labels)
    # (2) Proposed CBT-guided graph super-resolution
    c_degree = np.zeros((sz1, sz2))
    c_degree2 = np.zeros((sz1, sz2))
    c_closeness = np.zeros((sz1, sz2))
    c_betweenness = np.zeros((sz1, sz2))
    residual = np.zeros((len(train_data), len(train_data[1]), len(train_data[1])))

    for i in range(sz1):

        residual[i][:][:] = np.abs(train_data[i][:][:] - CBT) #residual brain graph
        G = nx.from_numpy_matrix(np.array(residual[i][:][:]))
        #c_degree2 = degrees(residual[i][:][:])
        for j in range(0, sz2):
             c_degree2[i][j] = degrees(residual[i][:][:])[j]
             c_degree[i][j] = G.degree(weight="weight")[j]
             #c_closeness[i][j] = closeness(G, residual[i][:][:])[j]
             #c_betweenness[i][j] = nx.betweenness_centrality(G, weight=True)[j]

    #print(c_degree)
    print(c_degree2)
    print(c_closeness)
    #print(c_betweenness)
