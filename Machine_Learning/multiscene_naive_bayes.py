# 导入必要库
import numpy as np
import re
import random
import operator
import feedparser




def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签
    return postingList, classVec




def createVocabList(dataSet):
    vocabSet = set()  # 创建空集合（自动去重）
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 求集合并集
    return sorted(list(vocabSet))  # 排序后返回




def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 初始化全0向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  # 词出现则置1
        else:
            print(f"警告：单词 '{word}' 不在词汇表中！")
    return returnVec





def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 初始化全0向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  # 词出现则计数+1
    return returnVec





def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 训练文档数量
    numWords = len(trainMatrix[0])  # 词汇表大小
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 侮辱性文档先验概率

    # 初始化概率（避免0概率导致乘积为0，使用拉普拉斯平滑）
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    # 统计各类别词出现次数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 侮辱性文档
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 正常文档
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 计算条件概率（取对数避免下溢出）
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive





def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 计算两类的后验概率（对数求和等价于原概率乘积）
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    return 1 if p1 > p0 else 0




def testingNB():
    # 1. 加载数据并创建词汇表
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)

    # 2. 构建训练集词向量矩阵
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    # 3. 训练分类器
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))

    # 4. 测试用例
    testEntries = [
        ['love', 'my', 'dalmation'],  # 正常
        ['stupid', 'garbage'],  # 侮辱性
        ['help', 'my', 'dog'],  # 正常
        ['worthless', 'stupid']  # 侮辱性
    ]

    # 5. 执行测试并输出结果
    print("=== 侮辱性留言分类测试 ===")
    for entry in testEntries:
        thisDoc = np.array(setOfWords2Vec(myVocabList, entry))
        classification = classifyNB(thisDoc, p0V, p1V, pAb)
        label = "侮辱性" if classification == 1 else "正常"
        print(f"输入：{entry} -> 分类结果：{label}")

    # 输出关键概率信息
    print(f"\n侮辱性文档先验概率：{pAb:.2f}")
    print(f"最具侮辱性特征词：{myVocabList[np.argmax(p1V)]}")





def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)  # 非字母数字分割
    # 过滤短词并转为小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]





def spamTest():
    docList = []
    classList = []
    fullText = []

    # 1. 加载邮件数据（spam=1，ham=0）
    for i in range(1, 26):
        # 加载垃圾邮件
        wordList = textParse(open('email/spam/%d.txt' % i, encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 加载正常邮件
        wordList = textParse(open('email/ham/%d.txt' % i, encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 2. 创建词汇表并构建训练集/测试集（留存交叉验证）
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))  # 50封邮件索引
    testSet = []

    # 随机选择10个作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    # 3. 训练分类器
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    # 4. 测试并计算错误率
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print(f"错分文档：{docList[docIndex]}")

    errorRate = float(errorCount) / len(testSet)
    print(f"\n=== 垃圾邮件分类测试 ===")
    print(f"测试集大小：{len(testSet)}")
    print(f"错误数量：{errorCount}")
    print(f"错误率：{errorRate:.2%}")
    return errorRate




def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in fullText:
        freqDict[token] = freqDict.get(token, 0) + 1  # 计数
    # 按频率降序排序
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))  # 取较短的RSS条目数

    # 1. 解析RSS内容
    for i in range(minLen):
        # 解析feed1（类别1）
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 解析feed0（类别0）
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 2. 创建词汇表（去除高频词，避免冗余）
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for word, _ in top30Words:
        if word in vocabList:
            vocabList.remove(word)  # 移除高频词

    # 3. 构建训练集/测试集
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):  # 随机选20个测试样本
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    # 4. 训练分类器
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pClass1 = trainNB0(np.array(trainMat), np.array(trainClasses))

    # 5. 测试并计算错误率
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pClass1) != classList[docIndex]:
            errorCount += 1

    errorRate = float(errorCount) / len(testSet)
    print(f"\n=== 地域广告分类测试 ===")
    print(f"测试集大小：{len(testSet)}")
    print(f"错误率：{errorRate:.2%}")
    return vocabList, p0V, p1V



def getTopWords(feed1, feed0):
    vocabList, p0V, p1V = localWords(feed1, feed0)
    top1 = []
    top0 = []

    # 筛选条件概率差异较大的词
    for i in range(len(p0V)):
        if p1V[i] > -6.0:  # 阈值可调整，过滤低概率词
            top1.append((vocabList[i], p1V[i]))
        if p0V[i] > -6.0:
            top0.append((vocabList[i], p0V[i]))

    # 按概率降序排序并输出
    sortedTop1 = sorted(top1, key=lambda x: x[1], reverse=True)
    sortedTop0 = sorted(top0, key=lambda x: x[1], reverse=True)

    print(f"\n=== 地域特征词汇 ===")
    print("Feed1（如纽约）特征词：")
    for word, prob in sortedTop1[:10]:  # 输出前10个
        print(f"  {word} (概率：{np.exp(prob):.4f})")  # 转换为原始概率

    print("\nFeed0（如旧金山）特征词：")
    for word, prob in sortedTop0[:10]:
        print(f"  {word} (概率：{np.exp(prob):.4f})")


"""
交互环境：实现文档中所有交互操作的自动化执行
"""


def interactiveMode():
    print("=" * 50)
    print("          朴素贝叶斯分类器交互环境          ")
    print("=" * 50)

    # 1. 执行侮辱性留言分类测试
    testingNB()

    # 2. 执行垃圾邮件分类测试（需提前准备email文件夹）
    try:
        spamTest()
    except FileNotFoundError:
        print("\n警告：未找到email文件夹，跳过垃圾邮件测试！")
        print("请在当前目录下创建email/spam和email/ham文件夹，并放入测试邮件。")

    # 3. 执行地域广告分类测试（需网络连接）
    try:
        print("\n正在获取RSS源数据...（请稍候）")

        ny = feedparser.parse('https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml')  # 对应原纽约源
        sf = feedparser.parse('https://www.sfchronicle.com/rss/feed/SF-Chronicle-Home-Headlines-421.php')  # 对应原旧金山源
        if len(ny['entries']) > 0 and len(sf['entries']) > 0:
            getTopWords(ny, sf)
        else:
            print("警告：RSS源获取失败，跳过地域分类测试！")
    except Exception as e:
        print(f"\n警告：地域分类测试出错：{str(e)}")

    print("\n" + "=" * 50)
    print("               交互操作执行完毕               ")
    print("=" * 50)


# 程序入口：运行交互环境
if __name__ == "__main__":
    interactiveMode()
