import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 导入sklearn库的数据清洗与分割函数，验证曲线函数
from sklearn.model_selection import ShuffleSplit, validation_curve
from sklearn.tree import DecisionTreeRegressor


# 定义模型在复杂度高的情况下性能表现的函数
# 随着模型复杂度的增加计算它的性能表现
def ModelComplexityPerformanceMetrics(x, y):

    # 清洗和分割数据对象定义
    # 参数1: n_splits表示重新清洗和分割数据的迭代次数，默认值为10
    # 参数2: test_size = 0.2表示有0.2的数据用于测试，也就是20%的数据用于测试，80%的数据用于训练。
    # 参数3: random_state表示随机数生成器的种子，如果希望第二次调用ShuffleSplit()方法的结果和第一次调用的结果一致，那么就可以设置一个值，多少都可以，生产环境不要设值
    #
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    # 定义从1到10为深度(max_depth)的参数值
    max_depth = np.arange(1, 11)
    # 通过不同的max_depth的参数值来计算训练集和测试集的分值
    # 参数1: 评估器，这里是决策树回归器
    # 参数2: 特征样本，房屋特征
    # 参数3: 目标标签，房屋价格
    # 参数4: 传入的深度参数名称
    # 参数5: 传入深度参数范围值
    # 参数6: 交叉验证生成器，或可迭代对象
    # 参数7: 评分器，是一个可调用对象
    #
    train_scores, test_scores = validation_curve(DecisionTreeRegressor(), x, y, param_name = "max_depth", param_range = max_depth, cv = cv, scoring = "r2")
    # 计算训练集分值和测试集分值的均值
    train_mean = np.mean(train_scores, axis = 1)
    test_mean = np.mean(test_scores, axis = 1)

    # 计算训练集分值和测试集分值的标准差
    train_std = np.std(train_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)
    # 绘制验证分值的曲线图
    # figsize表示要绘制的图形窗口大小，单位是英寸
    plt.figure(figsize=(7, 5))
    # 在绘制的图形窗口上添加一个标题
    plt.title("Decision Tree Regressor Complexity Performance")
    # 绘制训练得分线，plot()方法
    # 参数1: x轴方向的值
    # 参数2: y轴方向的值
    # 参数3: 绘制出来的线的风格，比如“o”表示一个圆点标记，“-”表示实线
    # 参数4: 绘制的线的颜色
    # 参数5: 图例上的标题
    plt.plot(max_depth, train_mean, "o-", color = "r", label = "Training Score")
    # 绘制测试得分线
    plt.plot(max_depth, test_mean, "o-", color = "g", label = "Validation Score")
    # plt.fill_between()方法表示为两条曲线描边，第一条是训练得分线，第二条是测试得分线
    # 参数1: x轴方向的值
    # 参数2: y轴方向的覆盖下限
    # 参数3：y轴方向的覆盖上限
    # 参数4: 设置覆盖区域的透明度
    # 参数5: 设置覆盖区域的颜色
    plt.fill_between(max_depth, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = "r")
    plt.fill_between(max_depth, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = "g")
    # 图上加标题注解
    # 添加图例
    plt.legend(loc = "lower right")
    # 添加x轴标题
    plt.xlabel("Maximum Depth")
    # 添加y轴标题
    plt.ylabel("Score")
    # 设置y轴方向的最小值和最大值
    plt.ylim([-0.05, 1.05])
    # 显示绘图
    plt.show()

data = pd.read_csv('../../../MyBook/Chapter-1-Housing-Price-Prediction/housing.csv')

# acquire the housing prices
prices = data["MEDV"]
# acquire the features of house
features = data.drop('MEDV', axis=1)
    
ModelComplexityPerformanceMetrics(features, prices)
