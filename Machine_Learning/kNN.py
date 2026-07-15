import numpy as np
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 使用np.tile
    sqDiffMat = diffMat **2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    # 使用with语句安全打开文件
    with open(filename, 'r') as fr:
        arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.50  # 保留50%作为测试集
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(
            normMat[i, :], 
            normMat[numTestVecs:m, :], 
            datingLabels[numTestVecs:m], 
            
        )
        print(f"分类器预测结果: {classifierResult}, 实际结果: {datingLabels[i]}")  # 使用f-string更清晰
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print(f"总错误率: {errorCount / float(numTestVecs)}")
    print(f"错误总数: {int(errorCount)}")


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games? "))
    ffMiles = float(input("frequent flier miles earned per year? "))
    iceCream = float(input("liters of ice cream consumed per year? "))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0(
        (inArr - minVals) / ranges,
        normMat,
        datingLabels,
        3
    )
    print(f"You will probably like this person: {resultList[classifierResult - 1]}")  # 使用f-string


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename, 'r') as fr:  # 使用with语句
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(f'trainingDigits/{fileNameStr}')  # 使用f-string
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(f'testDigits/{fileNameStr}')  # 使用f-string
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(f"分类器预测结果: {classifierResult}, 实际结果: {classNumStr}")  # 使用f-string
        if classifierResult != classNumStr:
            errorCount += 1.0
    print(f"\n错误总数: {int(errorCount)}")
    print(f"\n总错误率: {errorCount / float(mTest)}")


# 测试入口（可选）
if __name__ == "__main__":
    # 可根据需要注释/取消注释以下测试函数
    # datingClassTest()    # 测试约会网站分类器
    # classifyPerson()     # 交互式预测喜欢程度
    # handwritingClassTest()  # 测试手写数字识别
Pass
