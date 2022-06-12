# 重新构建模型
model = build_model()

# 设置早期停止，如果20次的迭代依旧没有降低验证损失，则自动停止训练
early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20)

# 重新训练模型，此时的callbacks有两个回调函数，所以使用数组的形式传入
history = model.fit(train_data, train_labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])

# 打印输出历史记录的曲线图
PlotHistory(history)
