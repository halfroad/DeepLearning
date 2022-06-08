import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

features = pd.read_csv("../../../MyBook/Chapter-2-Titanic-Survival-Exploration/titanic_dataset.csv")

y_train = features["Survived"]
X_train = features.drop("Survived", axis = 1)

head = X_train.head()

print(head)
print("X_train.shape = {}, y_train.shape= {}".format(X_train.shape, y_train.shape))

X_train.info()
print(X_train.isnull().sum())

sb.histplot(X_train["Age"].dropna(), kde = True)

plt.title("Ages Distribution")
plt.show()