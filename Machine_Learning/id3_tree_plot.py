import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 决策节点（判断节点）的样式
leafNode = dict(boxstyle="round4", fc="0.8")  # 叶子节点的样式
arrow_args = dict(arrowstyle="<-")  # 箭头的样式


def plotNode(nodeTxt, centerPt, parentPt, nodeType):

    #绘制带箭头的注解

    # createPlot.ax1 是在 createPlot 函数中定义的全局绘图区
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def getNumLeafs(myTree):
    """
    获取决策树的叶子节点数目
    """
    numLeafs = 0
    # Python 3 转换: .keys() 返回一个视图，需要转为 list 才能索引
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():

        if isinstance(secondDict[key], dict):  # 测试节点的数据类型是否为字典
            numLeafs += getNumLeafs(secondDict[key])  # 递归调用
        else:
            numLeafs += 1  # 是叶子节点
    return numLeafs


def getTreeDepth(myTree):
    """
    获取决策树的深度（层数）
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():

        if isinstance(secondDict[key], dict):  # 如果是判断节点
            thisDepth = 1 + getTreeDepth(secondDict[key])  # 递归, 深度+1
        else:
            thisDepth = 1  # 叶子节点深度为 1

        if thisDepth > maxDepth:
            maxDepth = thisDepth  # 更新最大深度
    return maxDepth


def retrieveTree(i):
    """
    返回预先存储的树信息，用于测试
    """
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},  # 树 0
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}  # 树 1
    ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子节点间填充文本信息
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center")


def plotTree(myTree, parentPt, nodeTxt):
    """
    递归绘制树形图
    """
    # 1. 计算宽与高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)

    # 2. 获取根节点名
    # Python 3 转换: list(keys())
    firstStr = list(myTree.keys())[0]

    # 3. 计算当前节点的中心位置
    # cntrPt 是在所有叶子节点的中间
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalw, plotTree.yOff)

    # 4. 绘制分支文本
    plotMidText(cntrPt, parentPt, nodeTxt)

    # 5. 绘制节点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)

    # 6. 获取子树
    secondDict = myTree[firstStr]

    # 7. 减少y偏移，为绘制下一层做准备
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalw

    # 8. 递归绘制所有子节点
    for key in secondDict.keys():
        # Python 3 转换: isinstance
        if isinstance(secondDict[key], dict):
            # 如果是决策节点，递归调用 plotTree
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalw
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

    # 9. 恢复y偏移
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalw


def createPlot(inTree):
    """
    创建绘图区，并调用 plotTree 来绘制决策树
    """
    # 创建一个新图形
    fig = plt.figure(facecolor='white')
    fig.clf()  # 清空绘图区

    # 定义绘图区属性，不显示x,y轴
    axprops = dict(xticks=[], yticks=[])

    # createPlot.ax1 定义为全局变量，供 plotNode 使用
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

    # 存储树的全局宽度和深度
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))

    # 存储全局的x, y偏移量
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0

    # 开始递归绘制
    plotTree(inTree, (0.5, 1.0), '')

    # 显示图形
    plt.show()


if __name__ == "__main__":

    tree_1 = retrieveTree(1)
    print(tree_1)

    myTree = retrieveTree(0)
    print("Tree 0:", myTree)

    num_leafs = getNumLeafs(myTree)
    tree_depth = getTreeDepth(myTree)

    print(f"叶子节点数 (NumLeafs): {num_leafs}")
    print(f"树的深度 (TreeDepth): {tree_depth}")
    createPlot(myTree)

    myTree['no surfacing'][3] = 'maybe'
    print("修改后的 Tree 0:", myTree)
createPlot(myTree)
