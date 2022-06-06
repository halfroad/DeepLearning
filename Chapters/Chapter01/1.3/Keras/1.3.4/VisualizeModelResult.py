import matplotlib.pyplot as plt
import numpy as np

# 绘制图来显示训练时的accuracy和loss

def PlotHistory(history):
    
    plt.figure()
    
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [1000$]")
    
    plt.plot(history.epoch, np.array(history.history["mae"]), label = "Train Loss")
    plt.plot(history.epoch, np.array(history.history["val_mae"]), label = "Val Loss")
    
    plt.legend()
    
    plt.ylim([0, 5])
    
    plt.show()
    