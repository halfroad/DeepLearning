import tensorflow as tf
import matplotlib.pyplot as plt

# 从TensorFlow导入Keras模块
from tensorflow import keras
import numpy as np

import sys
sys.path.insert(1, "../1.3.4")

from VisualizeModelResult import PlotHistory

# 自定义一个回调对象类，在每次epoch（代）结束时都会调用该函数
class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):

        if epoch % 100 == 0:
            print("")
        else:
            print(".", end = "")

EPOCHS = 500

# 加载波士顿房价数据集
(train_data, train_labels), (test_data, test_labels) = keras.datasets.boston_housing.load_data()

# 清洗训练集数据

# np.random.random() 表示返回在0.0到1.0之间指定个数的随机浮点数
# np.argsort() 表示返回对数组进行排序的索引

order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# 归一化处理数据
# 对不同的范围和比例进行归一化处理，并且每个元素都要减去均值后再除以标准差
# 虽然模型在没有特征归一化时也可以做到收敛，但是这会让训练更加困难
# 而且会导致结果模型依赖于训练数据

mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print("train_data.shape: {}, train_labels.shape: {}.".format(train_data.shape, train_labels.shape))
print("test_data.shape: {}, test_labels.shape: {}.".format(test_data.shape, test_labels.shape))

def PlotVersusPriceFigure(y_true_prices, y_predict_prices):
    
    # Create a window with dimension 10 * 7
    plt.figure(figsize = (10, 7))
    
    # Draw the true prices for figure 1
    X_show = np.rint(np.linspace(1, np.max(y_true_prices), len(y_true_prices))).astype(int)
    
    # Draw the line for figure 1, method plot():
    # Parameter 1: the values of X axis direction, the true prices from lowest to highest
    # Parameter 2: the values of Y axis direction, the true prices
    # Parameter 3: the style of the drawn line, i.e. "o-" means the circular dot, "-" means the solid line
    # Parameter 4: the color of the drawn line, here is cyan
    plt.plot(X_show, y_true_prices, "o-", color = "c")
    
    # Figure 2 is the predicted prices, be stacked over figure 1
    X_show_predicted = np.rint(np.linspace(1, np.max(y_predict_prices), len(y_predict_prices))).astype(int)
    
    # Draw the figure 2, method plot():
    # Parameter 1: the values of X axis direction, the predicted prices from lowest to highest
    # Parameter 2: the values of Y axis direction, the predicted prices
    # Parameter 3: the style of the drawn line, i.e. "o-" means the circular dot, "-"means the solid line
    # Parameter 4: The color of the drawn line, here is magenta
    plt.plot(X_show_predicted, y_predict_prices, "o-", color = "m")
    
    # Add title
    plt.title("Housing Prices Prediction")
    
    # Add legend
    plt.legend(loc="lower right", labels=["True Prices", "Predicted Prices"])
    
    # Add title for X axis
    plt.xlabel("House's Price Tendency By Array")
    
    # Add title for Y axis
    plt.ylabel("House's Price")
    
    # Show the plot
    plt.show()
    
# 定义创建模型函数
def build_model():

    model = keras.Sequential([
        keras.layers.Dense(64, activation = tf.nn.relu, input_shape = (train_data.shape[1],)),
        keras.layers.Dense(64, activation = tf.nn.relu),
        keras.layers.Dense(1)
        ])

    # 使用RMSProp（均方根传播）优化器，他可以加速梯度下降，其中学习速度适用于每个参数
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    # mse (均方差)一般用于回归问题的损失函数
    # mae (平均绝对误差)一般用于回归问题的测量/评估
    model.compile(loss = "mse", optimizer = optimizer, metrics = ["mae"])
    
    return model

model = build_model()

# 查看模型的架构
model.summary()

# 训练模型
# 参数1: 房屋特征数据
# 参数2: 房屋价格数据
# 参数3: 迭代次数
# 参数4: 验证集分割比例, 0.2表示20%的数据用于验证，80%的数据用于训练
# 参数5: 输出打印日志信息，0表示不输出打印日志信息
# 参数6: 回调对象，这里使用自定义的回调类PrintDot
# 
history = model.fit(train_data, train_labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [PrintDot()])

PlotHistory(history)

# 重新构建模型
model = build_model()

# 设置早期停止，如果20次的迭代依旧没有降低验证损失，则自动停止训练
early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20)

# 重新训练模型，此时的callbacks有两个回调函数，所以使用数组的形式传入
history = model.fit(train_data, train_labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])

# 打印输出历史记录的曲线图
PlotHistory(history)


[loss, mae] = model.evaluate(test_data, test_labels, verbose = 0)
print("Testing set Mean Abs Error: ${: 7.2f}".format(mae * 1000))

# 使用测试数据集预测模型
test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [1000$]")
plt.ylabel("Predictions [1000$]")
plt.axis("Equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])

plt.show()

error = test_predictions - test_labels

plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
plt.ylabel("Count")

plt.show()

PlotVersusPriceFigure(test_labels, test_predictions)