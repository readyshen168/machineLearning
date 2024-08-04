import matplotlib.pyplot as plt
from kNN import *


# 画出约会样本图
def showDatingData():
    # 获取样本矩阵和类标签向量
    dDmat, dLabels = file2matirx('datingTestSet2.txt')
    fig, ax = plt.subplots()
    ax.scatter(dDmat[:, 1], dDmat[:, 2])
    plt.show()


showDatingData()
