from math import log


# 计算给定数据集的香农熵值
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 熵值：
    shannoEnt = 0.0
    for key in labelCounts:
        # 香农公式
        prob = float(labelCounts[key]) / numEntries
        shannoEnt -= prob * log(prob,2)
    # 返回熵值
    return shannoEnt
