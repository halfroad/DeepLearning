import tensorflow as tf

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