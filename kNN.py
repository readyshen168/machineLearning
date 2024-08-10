from numpy import *
import operator


def createDataSet():
    group = array(
        [[1.0, 1.1],
         [1.0, 1.0],
         [0, 0],
         [0, 0.1]
         ]
    )
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# k-邻近算法
def classif0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 从文本文件中解析数据
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    # 训练样本矩阵
    returnMat = zeros((numberOfLines, 3))
    # 类标签向量
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        # 填充训练样本矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 填充类标签向量
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回训练样本矩阵和类标签向量
    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    # 得出每一列的最小值和最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    normDataSet = zeros(dataSet.shape)
    # 数据集的行数
    rows = dataSet.shape[0]
    # 得出数据集每个数与其所在列最小值差值
    normDataSet = dataSet - tile(minVals, (rows, 1))
    # 再除以每一列最大值与最小值之差
    ranges = maxVals - minVals
    normDataSet = normDataSet / tile(ranges, (rows, 1))
    # 返回归一化数据集， 原始数据每一列最大值与最小值的差，每一列的最小值
    return normDataSet, ranges, minVals


# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRtio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRtio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classif0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
