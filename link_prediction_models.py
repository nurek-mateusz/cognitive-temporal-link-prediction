# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 21:57:32 2023

@author: duany1
"""

import numpy as np
import pandas as pd
import time
from os import walk
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
import sys
import warnings
import os
import re


warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_data(path):
    df = pd.read_csv(path, sep=';')
    df['EventDate'] = pd.to_datetime(df['EventDate'])
    df['EventDate'] = df['EventDate'].astype(int) / 10**9
    df.columns = ['uid1', 'uid2', 'time']
    df = df.sort_values('time', ascending=True)
    df = df[df.uid1 != df.uid2]
    df = df.drop_duplicates()
    return df


def load_cogsnet_vector(path):
    df_cogsnet_vector = pd.read_csv(path, converters={"CogsnetVector": literal_eval})
    cogsnet_vector = df_cogsnet_vector.values.tolist()
    return cogsnet_vector


def load_cogsnet_weights(path):
    df_weights = pd.read_csv(path, sep=',')
    return df_weights
    

def Data_Shape(Data):
    #counts of links, nodes and timestamps
    MaxLinkNum = Data.shape[0]
    
    List_node = []
    for row in range(MaxLinkNum):
        List_node.append(Data[row][0])
        List_node.append(Data[row][1])
    List_node = list(set(List_node))
    MaxNodeNum =  len(List_node)
    
    List_time = []
    for row in range(MaxLinkNum):
        List_time.append(Data[row][2])
    List_time = list(set(List_time))
    MaxTimeNum =  len(List_time)
    
    return MaxLinkNum, MaxNodeNum, MaxTimeNum, List_node, List_time


def MatrixA(MaxNodeNum, Data, convertedNodeIDs):
    #network adjacency matrix
    MatrixAdjacency = np.zeros([MaxNodeNum, MaxNodeNum])
    for col in range(Data.shape[0]):
        i = int(convertedNodeIDs[convertedNodeIDs.realID == int(Data[col][0])].convertedID)
        j = int(convertedNodeIDs[convertedNodeIDs.realID == int(Data[col][1])].convertedID)
        MatrixAdjacency[i, j] = int(MatrixAdjacency[i, j]) + 1
        MatrixAdjacency[j, i] = int(MatrixAdjacency[j, i]) + 1
    return MatrixAdjacency


def CN(MatrixAdjacency_Train):
    MatrixAdjacency_Train = spones_2D(MatrixAdjacency_Train)
    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train)
    return Matrix_similarity


def cogsnet(MatrixAdjacency_Cos, Matrix_similarity_CN, alpha=0.5):
    Matrix_similarity_cogsnet = (alpha * MatrixAdjacency_Cos * (1/max(map(max,MatrixAdjacency_Cos)))) + ((1-alpha) * Matrix_similarity_CN * (1/max(map(max,Matrix_similarity_CN))))
    return Matrix_similarity_cogsnet


def cogsnet_cn(MatrixAdjacency_Train, MaxNodeNum):
    Matrix_Train_Weights = np.zeros((MaxNodeNum, MaxNodeNum))
    edge_exists = MatrixAdjacency_Train > 0
    for i in range(MaxNodeNum):
        for j in range(i + 1, MaxNodeNum):
            common_neighbors = edge_exists[i] & edge_exists[j]
            Matrix_Train_Weights[i][j] = np.sum((MatrixAdjacency_Train[i] + MatrixAdjacency_Train[j]) * common_neighbors)
            Matrix_Train_Weights[j][i] = Matrix_Train_Weights[i][j]
    return Matrix_Train_Weights


def Cos_TimeVector(MaxNodeNum, vector, convertedNodeIDs):
    #for congsnet
    MatrixAdjacency_Cos = np.zeros([MaxNodeNum, MaxNodeNum]) 
    for i in range(len(vector)):
        for j in range(i + 1, len(vector)):
            a = int(convertedNodeIDs[convertedNodeIDs.realID == int(vector[i][0])].convertedID)
            b = int(convertedNodeIDs[convertedNodeIDs.realID == int(vector[j][0])].convertedID)
            MatrixAdjacency_Cos[a, b] = cosine_similarity([vector[i][1]], [vector[j][1]])
            MatrixAdjacency_Cos[b, a] = cosine_similarity([vector[i][1]], [vector[j][1]])
    return MatrixAdjacency_Cos


def spones_2D(Matrix):
    for i in range(len(Matrix)):
        for j in range(i + 1, len(Matrix)):
            if Matrix[i][j] != 0:
                Matrix[i][j] = 1
                Matrix[j][i] = 1
    return Matrix


def TimeVector(MaxNodeNum, MaxTimeNum, Data, convertedNodeIDs, convertedTimeIDs):
    Matrix_TimeVector =np.zeros([MaxNodeNum, MaxTimeNum])
    for col in range(Data.shape[0]):
        i = int(convertedNodeIDs[convertedNodeIDs.realID == int(Data[col][0])].convertedID)
        j = int(convertedNodeIDs[convertedNodeIDs.realID == int(Data[col][1])].convertedID)
        t = int(convertedTimeIDs[convertedTimeIDs.realID == int(Data[col][2])].convertedID)
        Matrix_TimeVector[i, t] = int(Matrix_TimeVector[i, t]) + 1
        Matrix_TimeVector[j, t] = int(Matrix_TimeVector[j, t]) + 1
    return Matrix_TimeVector


def Cos_TimeVector_2(MaxNodeNum, Matrix_TimeVector):
    #for NSTV
    MatrixAdjacency_Cos = np.zeros([MaxNodeNum, MaxNodeNum]) 
    for i in range(Matrix_TimeVector.shape[0]):
        for j in range(i + 1, Matrix_TimeVector.shape[0]):
            MatrixAdjacency_Cos[i, j] = cosine_similarity([Matrix_TimeVector[i]], [Matrix_TimeVector[j]])
            MatrixAdjacency_Cos[j, i] = cosine_similarity([Matrix_TimeVector[i]], [Matrix_TimeVector[j]])
    return MatrixAdjacency_Cos


def NSTV(MatrixAdjacency_Cos, Matrix_sim, alpha=0.5):
    Matrix_similarity_NSTV = (alpha * MatrixAdjacency_Cos * (1/max(map(max,MatrixAdjacency_Cos)))) + ((1-alpha) * Matrix_sim * (1/max(map(max,Matrix_sim))))
    return Matrix_similarity_NSTV


def Calculation_AUC(Matrix_similarity, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum):
    AUCnum = 672400

    #similarity of non-exist links and test links
    Matrix_similarity = np.triu(Matrix_similarity - Matrix_similarity * spones_2D(MatrixAdjacency_Train))
    #index of non-exist links
    Matrix_NoExist = np.ones([MaxNodeNum, MaxNodeNum]) - spones_2D(MatrixAdjacency_Train) - spones_2D(MatrixAdjacency_Test) - np.eye(MaxNodeNum)

    #upper triangular matrix of Test Matrix and non-exist matirx
    Test = np.triu(MatrixAdjacency_Test)
    NoExist = np.triu(Matrix_NoExist)

    #counts of test links and non-exist links
    Test_num = np.count_nonzero(Test)
    NoExist_num = np.count_nonzero(NoExist)

    #'AUCnum' sampling comparisons between Test_rd[i] and NoExist_rd
    Test_rd = [int(x) for index,x in enumerate((Test_num * np.random.rand(1,AUCnum))[0])]
    NoExist_rd = [int(x) for index,x in enumerate((NoExist_num * np.random.rand(1,AUCnum))[0])]

    #similarity matrix of test set and non-exist set
    TestPre= Matrix_similarity * Test
    NoExistPre = Matrix_similarity * NoExist

    #similarity vector of test set and non-exist set
    TestIndex = np.argwhere(Test > 0)
    Test_Data = np.array([TestPre[x[0],x[1]] for index,x in enumerate(TestIndex)]).T
    NoExistIndex = np.argwhere(NoExist == 1)
    NoExist_Data = np.array([NoExistPre[x[0],x[1]] for index,x in enumerate(NoExistIndex)]).T

    Test_rd = np.array([Test_Data[x] for index,x in enumerate(Test_rd)])
    NoExist_rd = np.array([NoExist_Data[x] for index,x in enumerate(NoExist_rd)])
    #calculate AUC
    n1,n2 = 0,0
    for num in range(AUCnum):
        if Test_rd[num] > NoExist_rd[num]:
            n1 += 1
        elif Test_rd[num] == NoExist_rd[num]:
            n2 += 0.5
        else:
            n1 += 0
    auc = float(n1+n2)/AUCnum
    return auc


def CN_baseline(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity_CN, MaxNodeNum, convertedNodeIDs):    
    auc = Calculation_AUC(Matrix_similarity_CN, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum)
    prec = Calculation_Precision(Matrix_similarity_CN, MatrixAdjacency_Train, data_test, MaxNodeNum, convertedNodeIDs)

    return [auc, prec]


def NSTV_baseline(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity_NSTV, MaxNodeNum, convertedNodeIDs):
    auc = Calculation_AUC(Matrix_similarity_NSTV, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum)
    prec = Calculation_Precision(Matrix_similarity_NSTV, MatrixAdjacency_Train, data_test, MaxNodeNum, convertedNodeIDs)

    return [auc, prec]


def NSCV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity_CN, MaxNodeNum, convertedNodeIDs, vector_path, alpha):
    cogsnet_vector = load_cogsnet_vector(vector_path)
    MatrixAdjacency_Cos = Cos_TimeVector(MaxNodeNum, cogsnet_vector, convertedNodeIDs)
    Matrix_similarity_cogsnet = cogsnet(MatrixAdjacency_Cos, Matrix_similarity_CN, alpha)
    
    auc = Calculation_AUC(Matrix_similarity_cogsnet, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum)
    prec = Calculation_Precision(Matrix_similarity_cogsnet, MatrixAdjacency_Train, data_test, MaxNodeNum, convertedNodeIDs)

    return [auc, prec]


def NSCTV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity_CN, Matrix_TimeVector, MaxNodeNum, convertedNodeIDs, vector_path, alpha):
    cogsnet_vector = load_cogsnet_vector(vector_path)
    for i in range(0,len(cogsnet_vector)):
        convertedID = int(convertedNodeIDs.convertedID[convertedNodeIDs.realID == cogsnet_vector[i][0]])
        cogsnet_vector[i][1] = np.hstack((cogsnet_vector[i][1], Matrix_TimeVector[convertedID,:]))

    MatrixAdjacency_Cos = Cos_TimeVector(MaxNodeNum, cogsnet_vector, convertedNodeIDs)
    Matrix_similarity_cogsnet = cogsnet(MatrixAdjacency_Cos, Matrix_similarity_CN, alpha)
    
    auc = Calculation_AUC(Matrix_similarity_cogsnet, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum)
    prec = Calculation_Precision(Matrix_similarity_cogsnet, MatrixAdjacency_Train, data_test, MaxNodeNum, convertedNodeIDs)
   
    return [auc, prec]


def CNS(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum, convertedNodeIDs, weights_path):
    weights = load_cogsnet_weights(weights_path)
    MatrixAdjacency_Weights = np.zeros((MaxNodeNum, MaxNodeNum))
    for uid1, uid2, weight in weights.values:
        MatrixAdjacency_Weights[int(convertedNodeIDs.convertedID[convertedNodeIDs.realID == uid1])][int(convertedNodeIDs.convertedID[convertedNodeIDs.realID == uid2])] = weight

    Matrix_Train_Weights = cogsnet_cn(MatrixAdjacency_Weights, MaxNodeNum)
    
    auc = Calculation_AUC(Matrix_Train_Weights, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum)
    prec = Calculation_Precision(Matrix_Train_Weights, MatrixAdjacency_Train, data_test, MaxNodeNum, convertedNodeIDs)

    return [auc, prec]


def CNSCV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum, convertedNodeIDs, vector_path, weights_path, alpha):
    weights = load_cogsnet_weights(weights_path)
    MatrixAdjacency_Weights = np.zeros((MaxNodeNum, MaxNodeNum))
    for uid1, uid2, weight in weights.values:
        MatrixAdjacency_Weights[int(convertedNodeIDs.convertedID[convertedNodeIDs.realID == uid1])][int(convertedNodeIDs.convertedID[convertedNodeIDs.realID == uid2])] = weight

    cogsnet_vector = load_cogsnet_vector(vector_path)
    MatrixAdjacency_Cos = Cos_TimeVector(MaxNodeNum, cogsnet_vector, convertedNodeIDs)
    Matrix_Train_Weights = cogsnet_cn(MatrixAdjacency_Weights, MaxNodeNum)
    Matrix_similarity_cogsnet = cogsnet(MatrixAdjacency_Cos, Matrix_Train_Weights, alpha)
   
    auc = Calculation_AUC(Matrix_similarity_cogsnet, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum)
    prec = Calculation_Precision(Matrix_similarity_cogsnet, MatrixAdjacency_Train, data_test, MaxNodeNum, convertedNodeIDs)

    return [auc, prec]


def CNSTV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_TimeVector, MaxNodeNum, convertedNodeIDs, weights_path, alpha):    
    MatrixAdjacency_Cos = Cos_TimeVector_2(MaxNodeNum, Matrix_TimeVector)

    weights = load_cogsnet_weights(weights_path)
    MatrixAdjacency_Weights = np.zeros((MaxNodeNum, MaxNodeNum))
    for uid1, uid2, weight in weights.values:
        MatrixAdjacency_Weights[int(convertedNodeIDs.convertedID[convertedNodeIDs.realID == uid1])][int(convertedNodeIDs.convertedID[convertedNodeIDs.realID == uid2])] = weight
    Matrix_Train_Weights = cogsnet_cn(MatrixAdjacency_Weights, MaxNodeNum)

    Matrix_similarity_NSTV = NSTV(MatrixAdjacency_Cos, Matrix_Train_Weights, alpha)
    
    auc = Calculation_AUC(Matrix_similarity_NSTV, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum)
    prec = Calculation_Precision(Matrix_similarity_NSTV, MatrixAdjacency_Train, data_test, MaxNodeNum, convertedNodeIDs)
 
    return [auc, prec]


def CNSCTV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_TimeVector, MaxNodeNum, convertedNodeIDs, vector_path, weights_path, alpha):    
    cogsnet_vector = load_cogsnet_vector(vector_path)
    for i in range(0,len(cogsnet_vector)):
        convertedID = int(convertedNodeIDs.convertedID[convertedNodeIDs.realID == cogsnet_vector[i][0]])
        cogsnet_vector[i][1] = np.hstack((cogsnet_vector[i][1], Matrix_TimeVector[convertedID,:]))

    MatrixAdjacency_Cos = Cos_TimeVector(MaxNodeNum, cogsnet_vector, convertedNodeIDs)

    weights = load_cogsnet_weights(weights_path)
    MatrixAdjacency_Weights = np.zeros((MaxNodeNum, MaxNodeNum))
    for uid1, uid2, weight in weights.values:
        MatrixAdjacency_Weights[int(convertedNodeIDs.convertedID[convertedNodeIDs.realID == uid1])][int(convertedNodeIDs.convertedID[convertedNodeIDs.realID == uid2])] = weight
    Matrix_Train_Weights = cogsnet_cn(MatrixAdjacency_Weights, MaxNodeNum)

    Matrix_similarity_cogsnet = cogsnet(MatrixAdjacency_Cos, Matrix_Train_Weights, alpha)
    
    auc = Calculation_AUC(Matrix_similarity_cogsnet, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum)
    prec = Calculation_Precision(Matrix_similarity_cogsnet, MatrixAdjacency_Train, data_test, MaxNodeNum, convertedNodeIDs)
 
    return [auc, prec]


def MarixA2triple(MatrixAdjacency, MatrixAdjacency_test):
    data_triple = []
    for i in range(len(MatrixAdjacency)):
        for j in range(i + 1, len(MatrixAdjacency)):
            a = [i, j, MatrixAdjacency[i,j], int(MatrixAdjacency_test[i,j])]
            data_triple.append(a)
    return data_triple


def MatrixA_test(MaxNodeNum, Data_test, convertedNodeIDs):
    #network adjacency matrix
    MatrixAdjacency_test = np.zeros([MaxNodeNum, MaxNodeNum])
    for col in range(Data_test.shape[0]):
        i = int(convertedNodeIDs[convertedNodeIDs.realID == int(Data_test[col][0])].convertedID)
        j = int(convertedNodeIDs[convertedNodeIDs.realID == int(Data_test[col][1])].convertedID)
        MatrixAdjacency_test[i, j] = 1
        MatrixAdjacency_test[j, i] = 1
    return MatrixAdjacency_test


def Calculation_Precision(Matrix_similarity, MatrixAdjacency_Train, data_test, MaxNodeNum, convertedNodeIDs):
    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    # L = number of new link from test set
    MatrixAdjacency_test = MatrixA_test(MaxNodeNum, data_test, convertedNodeIDs)

    #similarity of non-exist links and test links
    Matrix_similarity = np.triu(Matrix_similarity - Matrix_similarity * spones_2D(MatrixAdjacency_Train))
    data_sim = MarixA2triple(Matrix_similarity, MatrixAdjacency_test)
    data_sim.sort(key=lambda x:x[2], reverse = True)

    n_newLinks = np.sum([1 if edgeTriple[3] == 1 else 0 for edgeTriple in data_sim])

    Precision = []
    for i in range(len(ratio)):
        L = int(n_newLinks * ratio[i])
        data_rank = data_sim[:L]
        m = 0
        for j in range(len(data_rank)):
            if data_rank[j][3] == 1:
                m = m + 1
        Precision.append(m / L)
    return Precision


if __name__ == "__main__":

    alpha = float(sys.argv[1])
    training_set_path = sys.argv[2]
    test_set_path = sys.argv[3]
    orginal_dataset = sys.argv[4]

    # LOAD DATA AND CREATE MATRICES
    data = load_data(orginal_dataset)
    data_numpy = data.to_numpy()
    
    MaxLinkNum, MaxNodeNum, MaxTimeNum, List_node, List_time = Data_Shape(data_numpy)
    
    convertedNodeIDs = pd.DataFrame({'realID': List_node, 'convertedID': list(range(0, len(List_node)))})
    convertedTimeIDs = pd.DataFrame({'realID': List_time, 'convertedID': list(range(0, len(List_time)))})
    
    data_train = np.loadtxt(training_set_path,skiprows=1,delimiter=';', dtype=int)
    MatrixAdjacency_Train = MatrixA(MaxNodeNum, data_train, convertedNodeIDs)
    data_test = np.loadtxt(test_set_path,skiprows=1,delimiter=';', dtype=int) 
    MatrixAdjacency_Test = MatrixA(MaxNodeNum, data_test, convertedNodeIDs)


    # COMPUTE CN
    Matrix_similarity_CN = CN(MatrixAdjacency_Train)


    # COMPUTE NSTV
    Matrix_TimeVector = TimeVector(MaxNodeNum, MaxTimeNum, data_train, convertedNodeIDs, convertedTimeIDs)
    MatrixAdjacency_Cos = Cos_TimeVector_2(MaxNodeNum, Matrix_TimeVector)
    Matrix_similarity_NSTV = NSTV(MatrixAdjacency_Cos, Matrix_similarity_CN)


    # COMPUTE BASELINE SCORES
    res = CN_baseline(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity_CN, MaxNodeNum, convertedNodeIDs)
    cn_auc = res[0]
    cn_prec = res[1]

    res = NSTV_baseline(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity_NSTV, MaxNodeNum, convertedNodeIDs)
    nstv_auc = res[0]
    nstv_prec = res[1]


    # COMPUTE COGSNET BASED METHODS
    
    # Extract params from file names
    pattern = re.compile(r'^(cogsnet-(exponential|power|linear)-\d+-\d+\.\d+-\d+\.\d+-\d+\.\d+-\d+)')
    files = set()

    for filename in os.listdir('results'):
        match = pattern.match(filename)
        if match:
            files.add(match.group(1))

    for params in files:
        aucs = []
        precs = []

        vector_sum = 'results/' + params + '-sum.csv'
        vector_avg = 'results/' + params + '-avg.csv'
        cogsnet_weights = 'results/' + params + '-weights.csv'

        # NSCV
        res = NSCV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity_CN, MaxNodeNum, convertedNodeIDs, vector_sum, alpha)
        aucs.append(['NSCV', vector_sum, res[0]])
        precs.append(['NSCV', vector_sum, res[1]])
        
        res = NSCV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity_CN, MaxNodeNum, convertedNodeIDs, vector_avg, alpha)
        aucs.append(['NSCV', vector_avg, res[0]])
        precs.append(['NSCV', vector_avg, res[1]])

        # CNSCV
        res = CNSCV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum, convertedNodeIDs, vector_sum, cogsnet_weights, alpha)
        aucs.append(['CNSCV', vector_sum, res[0]])
        precs.append(['CNSCV', vector_sum, res[1]])
        
        res = CNSCV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum, convertedNodeIDs, vector_avg, cogsnet_weights, alpha)
        aucs.append(['CNSCV', vector_avg, res[0]])
        precs.append(['CNSCV', vector_avg, res[1]])

        # NSCTV
        res = NSCTV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity_CN, Matrix_TimeVector, MaxNodeNum, convertedNodeIDs, vector_sum, alpha)
        aucs.append(['NSCTV', vector_sum, res[0]])
        precs.append(['NSCTV', vector_sum, res[1]])
        
        res = NSCTV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity_CN, Matrix_TimeVector, MaxNodeNum, convertedNodeIDs, vector_avg, alpha)
        aucs.append(['NSCTV', vector_avg, res[0]])
        precs.append(['NSCTV', vector_avg, res[1]])

        # CNSCTV
        res = CNSCTV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_TimeVector, MaxNodeNum, convertedNodeIDs, vector_sum, cogsnet_weights, alpha)
        aucs.append(['CNSCTV', vector_sum, res[0]])
        precs.append(['CNSCTV', vector_sum, res[1]])
        
        res = CNSCTV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_TimeVector, MaxNodeNum, convertedNodeIDs, vector_avg, cogsnet_weights, alpha)
        aucs.append(['CNSCTV', vector_avg, res[0]])
        precs.append(['CNSCTV', vector_avg, res[1]])

        # CNSTV
        res = CNSTV(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_TimeVector, MaxNodeNum, convertedNodeIDs, cogsnet_weights, alpha)
        aucs.append(['CNSTV', vector_sum, res[0]])
        precs.append(['CNSTV', vector_sum, res[1]])

        # CNS
        res = CNS(data_test, MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum, convertedNodeIDs, cogsnet_weights)
        aucs.append(['CNS', cogsnet_weights, res[0]])
        precs.append(['CNS', cogsnet_weights, res[1]])

        # Save results
        aucs.append(['CN', np.nan, cn_auc])
        precs.append(['CN', np.nan, cn_prec])
        aucs.append(['NSTV', np.nan, nstv_auc])
        precs.append(['NSTV', np.nan, nstv_prec])
        df_auc = pd.DataFrame(columns=['model','params','auc'], data=aucs)
        df_auc.to_csv('results/lp-' + params + '-auc.csv', index=False)
        df_prec = pd.DataFrame(columns=['model','params','precision'], data=precs)
        df_prec.to_csv('results/lp-' + params + '-prec.csv', index=False)















