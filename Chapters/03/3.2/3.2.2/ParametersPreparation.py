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
keepProbValue = 0.75
