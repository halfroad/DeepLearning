import pandas as pd
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# %matplotlib inline不被Thonny IDE支持，所以最后需要用plt.show()显示
# 导入sklearn的清洗分割、学习曲线和决策树回归的对象和函数
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
data = pd.read_csv('../../../MyBook/Chapter-1-Housing-Price-Prediction/housing.csv')

# acquire the housing prices
prices = data["MEDV"]

#acquire the features of house
features = data.drop('MEDV', axis=1)

# 定义模型的性能对比函数
# 通过不同大小的深度值来创建for循环里的模型，然后以图的形式展现

def ModelLearningGraphMetrics(x, y):
    # 清洗和分割数据对象定义
    # 参数1: n_splits表示重新清洗和分割数据的迭代次数，默认值为10
    # 参数2: test_size = 0.2表示有0.2的数据用于测试，80%的数据用于训练。
    # 参数3: random_state表示随机数生成器的种子，如果希望第二次调用ShuffleSplit()方法的结果和第一次调用的结果一致，那么就可以设置一个值，多少都可以，生产环境不要设值
    #
    #
    #
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    # 生成训练集大小
    # 函数np.rint()是计算数组各元素的四舍五入的值到最近的整数
    # 函数np.linspace(start_i, strop_i, num)表示从起始值到结束值之间，以均匀的间隔返回指定个数的值。
    # 此处就是从1开始，以x结束的总行数80%的数据，数据间隔是9，最后将数据元素都转换成整数
    train_sizes = np.rint(np.linspace(1, x.shape[0] * 0.8 - 1, 9)).astype(int)
    # 创建一个窗口，大小为10 * 7, 单位是英寸inch
    fig = plt.figure(figsize = (10, 7))
    # 根据深度值创建不同的模型
    # 这里的深度值就是1, 3, 6, 10这四个
    for k, depth in enumerate([1, 3, 6, 10]):
        # 根据深度(max_depth)值来创建决策树回归器
        regressor = DecisionTreeRegressor(max_depth = depth)
        # 通过学习曲线函数计算训练集和测试集的分值
        # 参数1: 评估器，这里就是决策树回归器
        # 参数2: 特征样本，房屋特征
        # 参数3: 目标标签，房屋价格
        # 参数4: 训练样本的个数，这里用来省城学习曲线的
        # 参数5: 交叉验证生成器，或可迭代对象
        # 参数6: 评分器，是一个可调用对象
        sizes, train_scores, test_scores = learning_curve(regressor, x, y, train_sizes = train_sizes, cv = cv, scoring = 'r2')
        # 计算训练集分值和测试集分值的标准差
        train_std = np.std(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        # 计算训练集分值和测试集分值的均值
        train_mean = np.mean(train_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)
        # 根据学期曲线值来绘制图，四个图的位置通过k + 1来控制
        ax = fig.add_subplot(2, 2, k + 1)
        # 绘制训练得分线，plot()方法
        # 参数1: x轴方向的值
        # 参数2: y轴方向的值
        # 参数3: 绘制出来的线的样式风格，比如o表示一个圆点标记，-表示实线
        # 参数4: 设置覆盖区域的透明度
        #  参数5: 覆盖区域的颜色
        ax.plot(sizes, train_mean, "o-", color = "r", label = "Training Scores")
        # 绘制测试得分线
        ax.plot(sizes, test_mean, "o-", color = "g", label = "Testing Score")
        # fill_between()方法表示为测试得分线描边
        # 参数1: x轴方向的值
        # 参数2: y轴方向的覆盖下限
        # 参数3: y轴方向的覆盖上限
        # 参数4: 设置覆盖区域的透明度
        # 参数5: 设置覆盖区域的颜色
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = "r")
        # fill_between()方法表示为测试得分线描边
        ax.fill_between(sizes, test_mean + test_std, alpha = 0.15, color = "g")
        # 在绘图的窗口上添加标题
        ax.set_title("max_depth = {}".format(depth))
        # 设置x轴的标题
        ax.set_xlabel("Number of Training Points")
        # 设置y轴的标题
        ax.set_ylabel("Score")
        # 设置x轴方向的最小值和最大值
        ax.set_xlim([0, x.shape[0] * 0.8])
        # 设置y轴方向的最小值和最大值
        ax.set_ylim([-0.05, 1.05])

    # 添加图例
    ax.legend(bbox_to_anchor = (1.05, 2.05), loc = "lower left", borderaxespad = 0.)
    # 添加图形总标题
    fig.suptitle("Decision Tree Regressor Learning Performance", fontsize = 16, y = 1.03)
    # 自动调整subplot复合图的区域的参数的布局。生产环境中不要使用该函数，因为这是一个实验特征性函数
    fig.tight_layout()
    # 显示绘图
    fig.show()

    plt.show()

ModelLearningGraphMetrics(features, prices)
