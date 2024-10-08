from os import listdir

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


# 约会网站预测数据
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classif0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


# 将图像转为向量
def img2vector(filename):
    # 最后返回的矩阵，把32*32矩阵转换成1*1024
    returnVect = zeros((1, 1024))
    fr = open(filename)
    # 循环矩阵每一行
    for i in range(32):
        # 读取某一行
        lineStr = fr.readline()
        # 把每一列数据复制给returnVect
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 测试算法： 使用K-近邻算法识别手写数字
def handwritingClassTest():
    # 训练文件夹中的文件清单
    trainingFileList = listdir('machinelearninginaction/Ch02/digits/trainingDigits')
    # 训练文件夹中文件的个数
    m = len(trainingFileList)
    # 训练矩阵和分类标签矩阵
    trainingMat = zeros((m, 1024))
    # hwLabels = zeros((0, m))
    hwLabels = []

    # 循环训练文件夹中每一个文件
    for i in range(m):
        # 每一个文件的路径
        trainingFile = 'machinelearninginaction/Ch02/digits/trainingDigits/' + trainingFileList[i]
        # 使用img2vector(trainingFile)方法向trainigMat中写入每一个文件的数据
        trainingMat[i:] = img2vector(trainingFile)
        # 向hwLabels中写入每一个文件名的第一个字符，即训练向量对应的标签结果
        # hwLabels[0, i] = int(trainingFileList[i].split('_')[0])
        hwLabels.append(int(trainingFileList[i].split('.')[0].split('_')[0]))

    # 记录错误次数的变量
    errorCount = 0.0
    # 测试矩阵的目录清单
    testFileList = listdir('machinelearninginaction/Ch02/digits/testDigits')
    # 测试向量的个数，即测试矩阵的行数
    mTest = len(testFileList)
    # 遍历每一个测试文件，计算每一个测试向量的classifierResult
    for j in range(mTest):
        # 测试矩阵提前给定的数字
        classNumStr = int(testFileList[j].split('_')[0])
        # 每一个测试文件的路径
        testFileName = 'machinelearninginaction/Ch02/digits/testDigits/' + testFileList[j]
        # 使用img2vector(testFileName)返回测试矩阵某行的向量
        vectorUnderTest = img2vector(testFileName)
        # 使用方法classif0(vectorUnderTest, trainingMat, hwLabels , 3)返回测试矩阵计算出来的数字classifierResult
        classifierResult = classif0(vectorUnderTest, trainingMat, hwLabels, 3)
        # 对比classifierResult和测试矩阵提前给定的数字classNumStr,若不相等，则errorCount加1
        if classifierResult != classNumStr:
            errorCount += 1
            # 打印错误结果
            print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is : %f" % (errorCount / float(mTest)))
