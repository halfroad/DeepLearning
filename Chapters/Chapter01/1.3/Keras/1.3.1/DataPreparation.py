import tensorflow as tf

# 从TensorFlow导入Keras模块
from tensorflow import keras
import numpy as np

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
