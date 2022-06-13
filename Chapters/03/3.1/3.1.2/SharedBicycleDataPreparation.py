import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import seaborn as sb

def Prepare():

	rides = pd.read_csv ( '../../../../MyBook/Chapter-3-Bike-Sharing-Analysis/hour.csv')
	head = rides.head()

	print(head)
	print(rides.shape)

	rides[: 24 * 10].plot (x = 'dteday',y = 'cnt')

	# 绘制的图的标注采用中文，可以使用如下方式
	# 变量font_path 是字体地址，读者使用自己的计算机或者服务器上的中文字体即可
	font_path = '../../../../MyBook/Songti.ttc'
	font = FontProperties(fname = font_path, size = "large", weight = "medium")

	# 创建两个绘图
	figure,(ax1,ax2) = plt.subplots(ncols = 2)
	# 设置两个绘图的总容器大小
	figure.set_size_inches (10, 5)

	# regplot()函数的参数说明如下
	# 参数1：×轴的值
	# 參数2：y轴的值
	# 参数3：原始数据
	# 參数4：绘图的对象
	# 参数5：绘图时每个点使用的标记符号

	# 显示温度的骑行数据
	# 温度值 = 温度值（概率）* 温度最大值41
	sb.regplot(x = rides["temp"] * 41, y = "cnt" , data = rides, ax = ax1, marker = "+")

	ax1.set_xlabel(u"温度", fontproperties = font)
	ax1.set_ylabel(u"骑行人数", fontproperties = font)

	# 显示体感温度的骑行数据
	# 实际体感温度值 = 体感温度值（概率）* 体感温度最大值 50
	sb.regplot(x = rides["atemp"] * 50, y = "cnt", data = rides, ax= ax2, marker = "^")

	ax2.set_xlabel(u"体感温度值", fontproperties = font)
	ax2.set_ylabel(u"骑行人数", fontproperties = font)

	# 创建两个绘图
	figure, (ax3, ax4) = plt.subplots(ncols=2)

	# 设置两个绘图的总容器大小
	figure.set_size_inches (10, 5)

	# 显示风速的骑行数据
	# 实际风速值 = 风速值（概率）* 风速最大值67
	sb.regplot(x = rides["windspeed"] * 67, y = "cnt", data = rides, ax = ax3, marker = "*")

	ax3.set_xlabel(u"实际风速值", fontproperties = font)
	ax3.set_ylabel(u"骑行人数", fontproperties = font)

	# 显示湿度的骑行数据
	# 实际湿度值 = 湿度值（概率）* 湿度最大值100
	sb.regplot(x = rides["hum"] * 100, y = "cnt", data = rides, ax = ax4, marker = ".")

	ax4.set_xlabel(u"实际湿度值", fontproperties = font)
	ax4.set_ylabel(u"骑行人数", fontproperties = font)

	# 温度、体感温度、临时骑行用户、已注册用户、湿度、风速和总骑行人数的相关性热图
	correlation = rides[["temp", "atemp", "casual", "registered", "hum", "windspeed", "cnt"]].corr()
	mask = np. array (correlation)

	# np.tril_indices_from（）表示返回数组的下三角形的索引
	mask[np.tril_indices_from(mask)] = False

	# 创建一个绘图
	figure, ax = plt.subplots()

	# 设置绘图的大小，20x10
	figure.set_size_inches (20, 10)

	# 用一个不彩色的小矩形组成一个大矩形
	# heatmap() 函数的参数说明
	# 参数1：二维的矩形数据集
	# 参数2：如果数据中有缺失值的 cell 就自动被屏蔽
	# 参数3：如果是Irue， 表示 cell 的宽和高相等
	# 参数 4：在每个 ce1l 上标出实际的数值
	sb.heatmap (correlation, mask = mask, square = True, annot = True)

	# 创建一个绘图
	figure, ax_by_month = plt.subplots(1)

	# 设置绘图的大小
	figure.set_size_inches (13, 7)

	# 对不同月份的骑行人数分组，然后计算均值
	# 通过reset index ()函数重置索引并创建一个新的 DataFrame 或者 Series 对象
	month_counts = pd.DataFrame(rides.groupby ("mnth") ["cnt"].mean()).reset_index()

	# 对DataPrame 根据月份从小到大排序，使用ascending 参数
	month_counts = month_counts.sort_values(by = "mnth", ascending = True)

	# 将月份数字转换成具体的月份字符串
	month_counts["mnth"] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

	# 绘制柱状图
	# 参数1：所有要绘制的数据
	# 参数2：×表示X轴的数据，数据字段名是mnth，数据在 data 里
	# 参数3：y表示y轴的数据，数据字段名是cnt，数据在 data 里
	# 参数4：被绘图的对象

	sb.barplot (data = month_counts, x = "mnth", y= "cnt", ax = ax_by_month)

	# 设置图的x轴标题
	ax_by_month.set_xlabel (u"月份", fontproperties = font)

	# 设置图的y轴标题
	ax_by_month.set_ylabel(u"骑行人数", fontproperties = font)

	# 设置图题
	ax_by_month.set_title(u"一年里每月平均骑行人数", fontproperties = font)

	# 创建一个绘图
	figure, ax_by_hour = plt.subplots (1)

	# 设置绘图的大小
	figure.set_size_inches (13, 7)

	# 将季节数字转换成具体的季节名称字符串
	ride_feature_copied = rides.copy()
	ride_feature_copied["season"] = ride_feature_copied.season.map({1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"})

	# 对各个季节一天里的每小时的骑行人数分组和排序，然后计算均值
	# 通过 reset_index()函数重置索引并创建一个新的 DataFrame 或者 Series 对象
	hour_counts = pd. DataFrame(ride_feature_copied.groupby(["hr", "season" ], sort = True)["cnt"].mean()).reset_index()

	# 绘制散点图
	# 参数1：所有要绘制的数据
	# 参数2：×表示 X轴的数据，数据字段名是hr，数据在data 里
	# 参数3：y表示y轴的数据，数据宇段名是cnt，数据在data 里
	# 参数4：根据指定字段的数据来绘制彩色的点
	# 参数 5：点与点之间是否绘制线来连接
	# 参数 6：被绘图的对象
	# 参数7：用不同的标记符号绘制不同的类别
	sb.pointplot(data = hour_counts, x = hour_counts["hr"], y = hour_counts["cnt"], hue = hour_counts["season" ], join = True, ax = ax_by_hour, markers = ["+", "o", "*", "^"])

	# 设置图的X轴标题
	ax_by_hour.set_xlabel(u"一天里的每时", fontproperties = font)

	# 设置图的Y轴标题
	ax_by_hour.set_ylabel(u"骑行人数", fontproperties = font)

	# 设罝图标题
	ax_by_hour.set_title(u"在四季里，以时为计数单位的每小时平均骑行人数", fontproperties = font)

	# 创建一个绘图

	figure, (ax_by_hour_weekday) = plt.subplots(1, 1)

	# 设置绘图大小
	figure.set_size_inches(13, 7)

	# 将工作日数字转换成对应的兴起及字符串
	ride_feature_copied["weekday"] = ride_feature_copied.weekday.map({0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"})

	# 对工作日和双休日各骑行人数分组和排序，然后计算均值
	# 通过reset_index()函数重制索引并创建一个新的DataFrame或者Series对象
	hour_weekday_counts = pd.DataFrame(ride_feature_copied.groupby(["hr", "weekday"], sort = True)["cnt"].mean()).reset_index()

	# 绘制散点图
	# 参数1: 所有要绘制的数据
	# 参数2: x表示X轴的数据，数据字段名是hour，数据在data里
	# 参数3: y表示Y轴的数据，数据字段名是cnt，数据在data里
	# 参数4: 根据指定字段的数据来绘制彩色的点
	# 参数5: 点与点之间是否绘制线来连接
	# 参数6: 被绘制的对象
	# 参数7: 用不同的标记符号绘制不同的类别

	sb.pointplot(data = hour_weekday_counts, x = hour_weekday_counts["hr"], y = hour_weekday_counts["cnt"], hue = hour_weekday_counts["weekday"], join = True, ax = ax_by_hour_weekday, markers = ["+", "o", "*", "^", "x", "h", "s"])

	# 设置图的X轴文字
	ax_by_hour_weekday.set_xlabel(u"一天里的每时", fontproperties = font)
	# 设置图的Y轴文字
	ax_by_hour_weekday.set_ylabel(u"骑行人数", fontproperties = font)
	# 设置图的标题
	ax_by_hour_weekday.set_title(u"在工作日和休息日，以时为技术单位的没下市平均骑行人数", fontproperties = font)

	# 创建一个绘图
	figure, ax_by_hour_casual_registered = plt.subplots(1)

	# 设置绘图大小
	figure.set_size_inches(13, 7)

	# pd.melt()方法表示从宽格式到长格式的转换数据，可选择设置标识符变量
	# 这里标识符字段是hr，变量字段是casual或者registered的新字段名variable，而值是value
	hour_casual_registered_data = pd.melt(rides[["hr", "casual", "registered"]], id_vars = ["hr"], value_vars = ["casual", "registered"])

	#然后通过reset_index()函数重置索引并创建一个新的DataFrame或Series对象
	# variable表示已注册/临时用户，而value表示骑行人数
	hour_casual_registered_data = pd.DataFrame(hour_casual_registered_data.groupby(["hr", "variable"], sort = True)["value"].mean()).reset_index()

	# 绘制散点图
	# 参数1: 所有要绘制的数据
	# 参数2: x表示X轴的数据，数据字段名是hr，数据在data里
	# 参数3: y表示Y轴的数据，数据字段名是value，数据在data里
	# 参数4: 会根据制定字段的数据来绘制彩色的点
	# 参数5: 绘制的色彩顺序
	# 参数6: 点与点之间是否绘制线来连接
	# 参数7: 被绘图的对象
	# 参数8: 用不同的标记符号绘制不同的类别
	#

	sb.pointplot(data = hour_casual_registered_data,
				 x = hour_casual_registered_data["hr"],
				 y = hour_casual_registered_data["value"],
				 hue = hour_casual_registered_data["variable"],
				 hue_order = ["casual", "registered"],
				 join = True,
				 ax = ax_by_hour_casual_registered,
				 markers = ["p", "s"])

	# 设置图的X轴标题
	ax_by_hour_casual_registered.set_xlabel(u"一天里的每时", fontproperties = font)
	# 设置图的Y轴标题
	ax_by_hour_casual_registered.set_ylabel(u"骑行人数", fontproperties = font)
	# 设置图的标题
	ax_by_hour_casual_registered.set_title(u"根据用户类型来计算每小时平均骑行人数", fontproperties = font)

	plt.show()
	
	return rides

Prepare()
