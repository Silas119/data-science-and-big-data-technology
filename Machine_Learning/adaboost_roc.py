import numpy as np
import matplotlib.pyplot as plt


def loadSimpData():
    """加载一个简单的二维数据集用于测试"""
    datMat = np.array([[1., 2.1],
                       [2., 1.1],
                       [1.3, 1.],
                       [1., 1.],
                       [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(fileName):
    """
    通用函数：解析以制表符分隔的浮点数数据文件
    每行最后一列为标签，其余为特征
    """
    with open(fileName) as fr:
        numFeat = len(fr.readline().split('\t'))  # 获取特征数量
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat - 1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    通过给定的阈值对数据进行分类
    :param dataMatrix: 输入数据矩阵 (m x n)
    :param dimen: 用于划分的特征维度索引
    :param threshVal: 阈值
    :param threshIneq: 不等号方向，'lt' 表示 <= 时为 -1，'gt' 表示 > 时为 -1
    :return: 分类结果向量 (m x 1)
    """
    retArray = np.ones((dataMatrix.shape[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    在加权数据集上构建最佳单层决策树（弱分类器）
    :param dataArr: 数据集 (m x n)
    :param classLabels: 类别标签 (长度为 m 的列表或数组)
    :param D: 数据权重向量 (m x 1)
    :return: 最佳决策树参数、最小加权错误率、最佳分类结果
    """
    dataMatrix = np.array(dataArr)
    labelMat = np.array(classLabels).reshape(-1, 1)  # 转为列向量
    m, n = dataMatrix.shape
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.zeros((m, 1))
    minError = np.inf  # 初始化最小错误率为正无穷

    for i in range(n):  # 遍历所有特征维度
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        # 遍历该维度上的所有可能阈值（包括边界外一点）
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:  # 尝试两种不等号方向
                threshVal = rangeMin + float(j) * stepSize
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.ones((m, 1))
                errArr[predictedVals == labelMat] = 0
                weightedError = np.dot(D.T, errArr)  # 加权错误率：D 是 (m,1)，errArr 是 (m,1)

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    AdaBoost 训练过程（使用单层决策树作为弱分类器）
    :param dataArr: 训练数据 (m x n)
    :param classLabels: 训练标签 (长度为 m)
    :param numIt: 最大迭代次数
    :return: 弱分类器列表、累计分类估计值 (m x 1)
    """
    weakClassArr = []
    dataMatrix = np.array(dataArr)
    m = dataMatrix.shape[0]
    D = np.ones((m, 1)) / m  # 初始化样本权重为均匀分布
    aggClassEst = np.zeros((m, 1))  # 累计分类器输出

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # 计算当前弱分类器的权重 alpha
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)

        # 更新样本权重 D
        labelMat = np.array(classLabels).reshape(-1, 1)
        expon = -1 * alpha * labelMat * classEst  # (m,1) 元素相乘
        D = D * np.exp(expon)
        D = D / D.sum()

        # 更新累计分类结果
        aggClassEst += alpha * classEst

        # 计算当前整体分类器的错误率
        aggErrors = np.sign(aggClassEst) != labelMat
        errorRate = np.mean(aggErrors)
        print("总错误率: ", errorRate)
        if errorRate == 0.0:
            break  # 如果错误率为0，提前终止训练
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    使用训练好的 AdaBoost 分类器对新数据进行分类
    :param datToClass: 待分类数据 (可以是列表、数组，支持多行)
    :param classifierArr: 弱分类器列表
    :return: 分类结果（+1 或 -1），形状 (m, 1)
    """
    dataMatrix = np.array(datToClass)
    if dataMatrix.ndim == 1:
        dataMatrix = dataMatrix.reshape(1, -1)
    m = dataMatrix.shape[0]
    aggClassEst = np.zeros((m, 1))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(
            dataMatrix,
            classifierArr[i]['dim'],
            classifierArr[i]['thresh'],
            classifierArr[i]['ineq']
        )
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    """
    绘制 ROC 曲线并计算 AUC（曲线下面积）
    :param predStrengths: 分类器输出的预测强度（未取 sign 的累计值），形状 (m, 1) 或 (m,)
    :param classLabels: 真实类别标签（列表或数组）
    """
    # 确保 predStrengths 是一维数组
    predStrengths = np.ravel(predStrengths)
    classLabels = np.array(classLabels)

    cur = (1.0, 1.0)  # 当前绘图点（起始于右上角）
    ySum = 0.0  # 用于计算 AUC
    numPosClas = np.sum(classLabels == 1.0)  # 正例数量
    yStep = 1.0 / numPosClas  # y轴步长（TPR 方向）
    xStep = 1.0 / (len(classLabels) - numPosClas)  # x轴步长（FPR 方向）

    # 按预测强度升序排序（从弱到强）
    sortedIndices = np.argsort(predStrengths)

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    # 遍历每个样本，绘制 ROC 曲线
    for index in sortedIndices:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]  # 累加矩形高度（用于 AUC）
        # 绘制线段
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)

    ax.plot([0, 1], [0, 1], 'b--')  # 对角线（随机猜测）
    plt.xlabel('假正率（False Positive Rate）')
    plt.ylabel('真正率（True Positive Rate）')
    plt.title('AdaBoost 马疝病检测系统的 ROC 曲线')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("曲线下面积（AUC）为: ", ySum * xStep)
