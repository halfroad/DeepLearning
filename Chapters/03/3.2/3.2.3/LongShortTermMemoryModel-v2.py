import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import sys
sys.path.insert(1, "../../3.1/3.1.4/")
from SharedBicycleDataRefinement import Refine

X_train, X_test, X_valid, y_train, y_test, y_valid = Refine()
#获取训练数据集列数
featuresNumber = X_train.shape[1]

# 预测目标的列数为3列，分别由cnt、casual、registered
targetsNumber = 3

# 小批次训练
batchSize = 10

# 每个批次时希望序列能记住的步长时100
stepsNumber = 100

# 设置LSTM Cell但愿的大小为256
lstmSize = 256

# 设置两层LSTM
layersNumber = 2

# 学习率
learningRate = 0.0005

# 保留率
keepProb = 0.75



def Create():

    # 创建输入纸盒目标值的占位符，之后会动态地传递数据到TensorFlow的计算图中
    inputs = tf.compat.v1.placeholder(tf.float32, [batchSize, None, featuresNumber], name = "inputs")
    targets = tf.compat.v1.placeholder(tf.float32, [batchSize, None, targetsNumber], name = "targets")

    # 创建保留率的占位符
    keepProb = tf.compat.v1.placeholder(tf.float32, name = "keepProb")

    # 创建学习率的占位符
    learningRate = tf.compat.v1.placeholder(tf.float32, name = "learningRate")

    # 添加命名范围，相当于给计算图上的tensor的RNN层添加一个前缀RNN_layers
    with tf.compat.v1.name_scope("RNN_layers"):
        # 参数state_is_tuple等于True表示接收并返回状态，状态是N维数组
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([CreateBasicCell() for _ in range(layersNumber)], state_is_tuple = True)
        # 初始化cell状态
        initial_state = cell.zero_state(batchSize, tf.float32)

    # 创建由RNNCell指定的循环神经网络，执行完全动态的输入展开
    outputs, final_state = tf.compat.v1.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)

    # 添加全连接输出层，输出层数为3，activation_fn设置为None表示使用线性激活，默认激活函数时ReLU
    predictions = tf.contrib.layers.fully_connected(outputs, 3, activation_fn = None)

    # 使用方差计算损失函数值
    cost = tf.compat.v1.losses.mean_squared_error(targets, predictions)

    # 设置优化器为Adam
    optimizer = tf.compat.v1.train.AdamOptimizer(learningRate).minimize(cost)

    # 计算验证精确度
    correctPredictions = tf.equal(tf.cast(tf.round(predictions), tf.int32), tf.cast(tf.round(targets), tf.int32))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correctPredictions, tf.float32))


# 定义创建LSTM的单元函数
def CreateBasicCell():

    # 创建基础LSTM Cell
    lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(lstmSize, reuse = tf.compat.v1.get_variable_scope().reuse)

    #添加dropout层到cell上
    return tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstmSize, output_keep_prob = keepProb)


Create()
