import numpy as np
from numpy import array, zeros, exp, random, nonzero, sign, dot, shape
from time import sleep
from os import listdir
import os


# SMO 辅助函数

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 简化版 SMO

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 转为 ndarray (m, n)
    dataMatrix = array(dataMatIn, dtype=float)      # shape: (m, n)
    labelMat = array(classLabels, dtype=float)      # shape: (m,)
    b = 0.0
    m, n = dataMatrix.shape
    alphas = zeros(m)
    iter_num = 0

    while iter_num < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # fXi = Σ α_j * y_j * (x_i · x_j) + b
            fXi = sum(alphas * labelMat * (dataMatrix @ dataMatrix[i])) + b
            Ei = fXi - labelMat[i]

            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
               ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):

                j = selectJrand(i, m)
                fXj = sum(alphas * labelMat * (dataMatrix @ dataMatrix[j])) + b
                Ej = fXj - labelMat[j]

                alphaIold = alphas[i]
                alphaJold = alphas[j]

                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    print("L==H")
                    continue

                # eta = 2*x_i·x_j - x_i·x_i - x_j·x_j
                eta = 2.0 * dot(dataMatrix[i], dataMatrix[j]) \
                      - dot(dataMatrix[i], dataMatrix[i]) \
                      - dot(dataMatrix[j], dataMatrix[j])

                if eta >= 0:
                    print("eta>=0")
                    continue

                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if abs(alphas[j] - alphaJold) < 1e-5:
                    print("j not moving enough")
                    continue

                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                # 更新 b
                b1 = b - Ei \
                     - labelMat[i] * (alphas[i] - alphaIold) * dot(dataMatrix[i], dataMatrix[i]) \
                     - labelMat[j] * (alphas[j] - alphaJold) * dot(dataMatrix[i], dataMatrix[j])
                b2 = b - Ej \
                     - labelMat[i] * (alphas[i] - alphaIold) * dot(dataMatrix[i], dataMatrix[j]) \
                     - labelMat[j] * (alphas[j] - alphaJold) * dot(dataMatrix[j], dataMatrix[j])

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1
                print(f"iter: {iter_num} i:{i}, pairs changed {alphaPairsChanged}")

        if alphaPairsChanged == 0:
            iter_num += 1
        else:
            iter_num = 0
        print(f"iteration number: {iter_num}")

    return b, alphas


#  核函数

def kernelTrans(X, A, kTup):
    """
    X: (m, n), A: (n,) → 返回 (m,) 的核向量
    """
    m, n = X.shape
    if kTup[0] == 'lin':
        return X @ A  # linear kernel
    elif kTup[0] == 'rbf':
        K = zeros(m)
        for j in range(m):
            delta = X[j] - A
            K[j] = dot(delta, delta)
        return exp(-K / (kTup[1] ** 2))
    else:
        raise NameError('That Kernel is not recognized')


# 完整版 SMO 所需结构

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = array(dataMatIn, dtype=float)       # (m, n)
        self.labelMat = array(classLabels, dtype=float)  # (m,)
        self.C = C
        self.tol = toler
        self.m = self.X.shape[0]
        self.alphas = zeros(self.m)
        self.b = 0.0
        self.eCache = zeros((self.m, 2))  # [valid, Ek]
        # Pre-compute kernel matrix
        self.K = zeros((self.m, self.m))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i], kTup)


def calcEk(oS, k):
    fXk = sum(oS.alphas * oS.labelMat * oS.K[:, k]) + oS.b
    Ek = fXk - oS.labelMat[k]
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0])[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
       ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i]
        alphaJold = oS.alphas[j]

        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])

        if L == H:
            print("L==H")
            return 0

        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0

        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)

        if abs(oS.alphas[j] - alphaJold) < 1e-5:
            print("j not moving enough")
            return 0

        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)

        b1 = oS.b - Ei \
             - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] \
             - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej \
             - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] \
             - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]

        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0

        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(dataMatIn, classLabels, C, toler, kTup)
    iter_num = 0
    entireSet = True
    alphaPairsChanged = 0

    while (iter_num < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print(f"fullSet, iter: {iter_num} i:{i}, pairs changed {alphaPairsChanged}")
            iter_num += 1
        else:
            nonBoundIs = nonzero((oS.alphas > 0) & (oS.alphas < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print(f"non-bound, iter: {iter_num} i:{i}, pairs changed {alphaPairsChanged}")
            iter_num += 1

        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True

        print(f"iteration number: {iter_num}")

    return oS.b, oS.alphas


#其他工具函数

def calcWs(alphas, dataArr, classLabels):
    X = array(dataArr)
    labelMat = array(classLabels)
    m, n = X.shape
    w = zeros(n)
    for i in range(m):
        w += alphas[i] * labelMat[i] * X[i]
    return w


def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = array(dataArr)
    labelMat = array(labelArr)
    svInd = nonzero(alphas > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print(f"there are {len(sVs)} Support Vectors")

    m = datMat.shape[0]
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i], ('rbf', k1))
        predict = sum(kernelEval * labelSV * alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print(f"the training error rate is: {errorCount / m:.6f}")

    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    datMat = array(dataArr)
    m = datMat.shape[0]
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i], ('rbf', k1))
        predict = sum(kernelEval * labelSV * alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print(f"the test error rate is: {errorCount / m:.6f}")


def img2vector(filename):
    returnVect = zeros(1024)
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[32 * i + j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(-1.0 if classNumStr == 9 else 1.0)
        fullPath = os.path.join(dirName, fileNameStr)
        trainingMat[i, :] = img2vector(fullPath)
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = array(dataArr)
    labelMat = array(labelArr)
    svInd = nonzero(alphas > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print(f"there are {len(sVs)} Support Vectors")

    m = datMat.shape[0]
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i], kTup)
        predict = sum(kernelEval * labelSV * alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print(f"the training error rate is: {errorCount / m:.6f}")

    dataArr, labelArr = loadImages('testDigits')
    datMat = array(dataArr)
    m = datMat.shape[0]
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i], kTup)
        predict = sum(kernelEval * labelSV * alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print(f"the test error rate is: {errorCount / m:.6f}")
