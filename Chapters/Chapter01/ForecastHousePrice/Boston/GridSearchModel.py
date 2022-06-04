import pandas as pd
# 从sklearn库导入网格搜索VC、数据清洗与分割、决策树和分值计算对象的函数
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):

    score = r2_score(y_true, y_predict)

    return score

# 定义网格搜索最佳模型函数
def GridSearchFitModel(x, y):
    # 清洗和分割数据对象定义，
    # 参数1: n_splits表示重新清洗和分割数据的迭代次数，默认值为10
    # 参数2: test_size = 0.2表示有0.2的数据用于测试，也就是20%的数据用于测试，80%的数据用于训练
    # 参数3: random_state表示随机数生成器的种子，如果希望第二次调用ShuffleSplit()方法的结果和第一次调用的结果一致，那么就可以设置一个值，多少都可以，生产环境不要设值
    #
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    # 创建决策树回归器对象
    regressor = DecisionTreeRegressor(random_state = 0)
    # 创建一个字典，表示max_depth的参数值是从1到10
    # 注意：如果代码运行的环境是Python 2，去掉这个list()函数调用
    params = {"max_depth": list(range(1, 10))}
    # 通过make_scorer()函数将上面定义的Peformance_metric()函数转换成计算分值函数
    scoring_fnc = make_scorer(score_func = performance_metric)
    # 创建网格搜索对象
    # 参数1: 评估器，就是回归器，这里表示的是决策树回归器
    # 参数2: 网格搜索参数
    # 参数3: 计算分值函数
    # 参数4: CV（Crossing-Validation）交叉验证，传入交叉验证生成器，或者可迭代对象
    grid = GridSearchCV(estimator = regressor, param_grid = params, scoring = scoring_fnc, cv = cv)
    # 根据数据计算/训练适合网格搜索对象的最佳模型
    grid = grid.fit(x, y)

    # 返回计算得到的最佳模型
    return grid.best_estimator_

data = pd.read_csv('../../../../MyBook/Chapter-1-Housing-Price-Prediction/housing.csv')

# acquire the housing prices
prices = data["MEDV"]

# acquire the features of house
features = data.drop('MEDV', axis=1)

x_train, x_test, y_train, y_test = train_test_split(features, prices, test_size=0.1, random_state=50)

# 网格搜索函数得到最佳模型
reg = GridSearchFitModel(x_train, y_train)

print("max_depth = {}".format(reg.get_params()["max_depth"]))
