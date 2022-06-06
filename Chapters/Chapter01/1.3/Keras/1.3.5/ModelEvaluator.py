[loss, mae] = model.evaluate(test_data, test_labels, verbose = 0)
print("Testing set Mean Abs Error: ${: 7.2f}".format(mae * 1000)
