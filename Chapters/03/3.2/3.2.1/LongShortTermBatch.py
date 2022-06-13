import numpy as np

def GetBatches(X, y, sequences, steps):

	# 使用每个序列大小乘以序列步骤得出每批次
	perBatch = sequences * steps

	# 用总数除以每批次得出需要多少个批次
	numberOfBatchs = len(X) / perBatch

	# 将X和y的值转换成numpy，然后去掉索引
	X = X.reset_index().values[:, 1:]
	Y = y.reset_index().values[:, 1:]

	# 取出最终的数据
	X, y = X: [numberOfBatches * perBatch], y[: numberOfBatches * perBatch]

	# 将X和y数据分别分割成steps个
	dataX = []
	dataY = []

	for i in range(0, numberOfBatches * perBatch, steps):

		dataX.append(x[i: i + steps])
		dataY.append(y[i: i + steps])

	# 将X和y数据转换成NDArray
	X = np.asarray(dataX)
	y = np.asarray(dataY)

	# 将X数据分割成[samples, time steps, features]的元素，并返回生成对象
	for i in range(0, (numberOfBatches * perBatch) / steps, sequences):

		# 使用yield关键字表示声称可迭代对象并返回
		yield X[i: i + sequances, :, :], y[i: i + sequences, :, :]
