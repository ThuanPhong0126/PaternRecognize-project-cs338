from matplotlib import colors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dt_train = pd.read_csv('./train.csv')
dt_dev = pd.read_csv('./dev.csv')
dt_test = pd.read_csv('./test.csv')

#Number of pair of questions
n_train = len(dt_train)
n_dev = len(dt_dev)
n_test = len(dt_test)

#Number of sample "0"
n_train0 = len(dt_train[dt_train['is_duplicate'] == 0])
n_dev0 = len(dt_dev[dt_dev['is_duplicate'] == 0])
n_test0 = len(dt_test[dt_test['is_duplicate'] == 0])
print("Số lượng dòng nhãn 0:", n_train0, n_dev0, n_test0)
print("Số lượng dòng nhãn 1:", n_train-n_train0, n_dev-n_dev0, n_test-n_test0)

l = 0
# Avg length of all questions in Train data
dt1 = dt_train['question1'].str.len()
dt2 = dt_train['question2'].str.len()
print("Độ dài trung bình câu hỏi trên tập Train:", (dt1.sum(axis = 0) + dt2.sum(axis = 0))/(n_train*2))
l += dt1.sum(axis = 0) + dt2.sum(axis = 0)

#Avg length of all questions in Dev data
dt1 = dt_dev['question1'].str.len()
dt2 = dt_dev['question2'].str.len()
print("Độ dài trung bình câu hỏi trên tập Dev:", (dt1.sum(axis = 0) + dt2.sum(axis = 0))/(n_dev*2))
l += dt1.sum(axis = 0) + dt2.sum(axis = 0)

#Avg length of all questions in Test data
dt1 = dt_test['question1'].str.len()
dt2 = dt_test['question2'].str.len()
print("Độ dài trung bình câu hỏi trên tập Test:", (dt1.sum(axis = 0) + dt2.sum(axis = 0))/(n_test*2))
l += dt1.sum(axis = 0) + dt2.sum(axis = 0)

#Avg length of all questions in whole data
print("Độ dài trung bình câu hỏi trên toàn bộ data:", l/(n_train*2 + n_dev*2 + n_test*2))


#Data Visualization
div_0 = [n_train0, n_dev0, n_test0]
div_1 = [n_train - n_train0, n_dev - n_dev0, n_test - n_test0]

ind = np.arange(len(div_0))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, div_0, width,
                label='Nhãn 0')
rects2 = ax.bar(ind + width/2, div_1, width,
                label='Nhãn 1')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Số lượng')
ax.set_xlabel('Tập data')
ax.set_title('Sơ đồ số lượng nhãn của bộ data')
ax.set_xticks(ind)
ax.set_xticklabels(("Train","Dev", "Test"))
ax.legend()

plt.show()