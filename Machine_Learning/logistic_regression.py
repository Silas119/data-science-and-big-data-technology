import numpy as np
import random
import os
import matplotlib.pyplot as plt

#Logistic回归梯度上升优化算法
def loadDataSet():
    """加载 testSet.txt 数据集"""
    dataMat = []
    labelMat = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'testSet.txt')
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    """Sigmoid 函数"""
    # 防止溢出：对大值做裁剪（可选但推荐）
    inX = np.clip(inX, -500, 500)
    return 1.0 / (1 + np.exp(-inX))

#转换NumPy矩阵数据类型
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.array(dataMatIn, dtype=np.float64)
    labelMat = np.array(classLabels, dtype=np.float64).reshape(-1, 1)
    m, n = dataMatrix.shape
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):#矩阵相乘
        h = sigmoid(dataMatrix @ weights)
        error = labelMat - h
        weights = weights + alpha * (dataMatrix.T @ error)
    return weights

#画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    # 自动处理 1D 或 2D 权重
    if weights.ndim == 1:
        w0, w1, w2 = weights[0], weights[1], weights[2]
    else:
        w0, w1, w2 = weights[0, 0], weights[1, 0], weights[2, 0]

    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = dataArr.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s', label='Class 1')
    ax.scatter(xcord2, ycord2, s=30, c='green', label='Class 0')

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-w0 - w1 * x) / w2
    ax.plot(x, y, 'b-', label='Decision Boundary')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    """随机梯度上升（单次遍历）"""
    dataMatrix = np.array(dataMatrix, dtype=np.float64)
    m, n = dataMatrix.shape
    weights = np.ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmoid(np.dot(dataMatrix[i], weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """改进的随机梯度上升（多次迭代 + 动态学习率）"""
    dataMatrix = np.array(dataMatrix, dtype=np.float64)
    m, n = dataMatrix.shape
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(np.dot(dataMatrix[randIndex], weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights


def classifyVector(inX, weights):
    """分类函数"""
    prob = sigmoid(np.dot(inX, weights))
    return 1.0 if prob > 0.5 else 0.0


def colicTest():
    """马疝病数据测试"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, 'horseColicTraining.txt')
    test_path = os.path.join(script_dir, 'horseColicTest.txt')

    trainingSet = []
    trainingLabels = []
    with open(train_path, 'r') as frTrain:
        for line in frTrain:
            currLine = line.strip().split('\t')
            if len(currLine) < 22:
                continue  # 跳过不完整行
            lineArr = [float(currLine[i]) for i in range(21)]
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))

    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, numIter=1000)

    errorCount = 0
    numTestVec = 0.0
    with open(test_path, 'r') as frTest:
        for line in frTest:
            currLine = line.strip().split('\t')
            if len(currLine) < 22:
                continue
            lineArr = [float(currLine[i]) for i in range(21)]
            if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
                errorCount += 1
            numTestVec += 1.0

    errorRate = errorCount / numTestVec if numTestVec > 0 else 0.0
    print("The error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    """多次测试取平均错误率"""
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    avg_error = errorSum / float(numTests)
    print("After %d iterations the average error rate is: %f" % (numTests, avg_error))
