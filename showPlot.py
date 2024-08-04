import matplotlib.pyplot as plt
from kNN import *


# 画出约会样本图
def showDatingData():
    # 获取样本矩阵和类标签向量
    dDmat, dLabels = file2matirx('datingTestSet2.txt')
    fig, ax = plt.subplots()
    # ax.scatter(dDmat[:, 1], dDmat[:, 2])
    ax.scatter(dDmat[:, 1], dDmat[:, 2], c=array(dLabels), s=15.0 * array(dLabels))
    # 使用标签向量作为颜色映射；点的大小与标签值成正比
    plt.show()


showDatingData()
