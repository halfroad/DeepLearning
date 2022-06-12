# 自定义一个回调对象类，在每次epoch（代）结束时都会调用该函数
class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):

        if epoch % 100 == 0:
            print("")
        else:
            print(".", end = "")

EPOCHS = 500

# 训练模型
# 参数1: 房屋特征数据
# 参数2: 房屋价格数据
# 参数3: 迭代次数
# 参数4: 验证集分割比例, 0.2表示20%的数据用于验证，80%的数据用于训练
# 参数5: 输出打印日志信息，0表示不输出打印日志信息
# 参数6: 回调对象，这里使用自定义的回调类PrintDot
# 
history = model.fit(train_data, train_labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [PrintDot()])
