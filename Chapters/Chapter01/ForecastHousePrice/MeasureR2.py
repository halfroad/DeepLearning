from sklearn.metrics import r2_score

test_y_true = [3, -0.5, 2, 7, 4.2]
test_y_predict = [2.5, 0.0, 2.1, 7.8, 5.3]

score = r2_score(test_y_true, test_y_predict)

print("Determined Coefficient, R^2 = {}".format(score))
