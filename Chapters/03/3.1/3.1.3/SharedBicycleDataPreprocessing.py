import pandas as pd

import sys
sys.path.insert(1, "../../3.1/3.1.2/")

from SharedBicycleDataPreparation import Prepare


def Preprocess():
	
	rides = Prepare()
	
	dummy_fields = ["season", "weathersit", "mnth", "hr", "weekday"]

	for each in dummy_fields:

		dummies = pd.get_dummies(rides[each], prefix = each, drop_first = False)
		rides = pd.concat([rides, dummies], axis = 1)

	fields_to_drop = ["instant", "season", "weathersit", "weekday", "atemp", "mnth", "workingday", "hr"]
	rides = rides.drop(fields_to_drop, axis = 1)

	scaled_features = {}
	quant_features = ["casual", "registered", "cnt", "temp", "hum", "windspeed"]

	for each in quant_features:

		# 计算这几个字段的均值和标准差
		mean, std = rides[each].mean(), rides[each].std()
		scaled_features[each] = [mean, std]

		# 数据减去均值后再除以标准差等于标准分值(Standard Score)，这样处理是为了使数据符合标准正态分布
		rides.loc[:, each] = (rides[each] - mean) / std

	# 分开features和target
	target_columns = ["cnt", "casual", "registered"]
	y_labels = rides[target_columns]

	# features不保留骑行人数字段
	X_features = rides.drop(target_columns, axis = 1)
	head = X_features.head()

	print(head)
	
	return X_features, y_labels

Preprocess()