"""
    随机森林：
        每次构建决策树模型时，不仅随机选择部分样本，而且还随机选择部分特征，
        这样的集合算法不仅规避了强势样本对预测结果的影响，而且也削弱了强制特征的影响，使模型的预测能力更加泛化
"""
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = []
with open("../data/bike_day.csv", 'r') as f:
    for line in f.readlines():
        data.append(line[:-1].split(","))  # 去除最后一个\n
# 整理输入和输出集(第一和第二行不需要参与训练)
day_header = np.array(data[0][2:13])
x = np.array(data[1:])[:, 2:13].astype("f8")
y = np.array(data[1:])[:, -1].astype("f8")
# 打乱数据集，拆分训练集和测试集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[:train_size], y[:train_size], y[:train_size]
# 训练随机森林模型
model = se.RandomForestRegressor(max_depth=10, n_estimators=100, min_samples_split=2)
# 输出预测结果R2得分
model.fit(x, y)
pred_test_y = model.predict(test_x)
print(f"bike_hour R2得分：{sm.r2_score(test_y, pred_test_y)}")
print(f"bike_hour 平均绝对值误差：{sm.mean_absolute_error(test_y, pred_test_y)}")
# 获取特征重要性
day_fi = model.feature_importances_

data = []
with open("../data/bike_hour.csv", 'r') as f:
    for line in f.readlines():
        data.append(line[:-1].split(","))  # 去除最后一个\n
# 整理输入和输出集(第一和第二行不需要参与训练)
hour_header = np.array(data[0][2:14])
x = np.array(data[1:])[:, 2:14].astype("f8")
y = np.array(data[1:])[:, -1].astype("f8")
# 打乱数据集，拆分训练集和测试集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[:train_size], y[:train_size], y[:train_size]
# 训练随机森林模型
model = se.RandomForestRegressor(max_depth=10, n_estimators=100, min_samples_split=2)
# 输出预测结果R2得分
model.fit(x, y)
pred_test_y = model.predict(test_x)
print(f"bike_hour R2得分：{sm.r2_score(test_y, pred_test_y)}")
print(f"bike_hour 平均绝对值误差：{sm.mean_absolute_error(test_y, pred_test_y)}")
# 获取特征重要性
hour_fi = model.feature_importances_

# 画图
mp.figure('Random Forest Regressor', facecolor='lightgray')
mp.subplot(2, 1, 1)  # 2行1列，第1个
mp.title('Day Feature Importance', fontsize=16)
mp.ylabel('Feature Importance', fontsize=14)
x = np.arange(day_fi.size)
# 对特征重要性进行排序，得到有序索引
sorted_indices = day_fi.argsort()[::-1]  # 倒排
day_fi = day_fi[sorted_indices]
mp.grid(linestyle=':')
# 对x轴刻度进行处理
mp.xticks(x, day_header[sorted_indices])
mp.bar(x, day_fi, 0.8, color='dodgerblue', label='Day Feature Importance')
mp.legend()
mp.tight_layout()
mp.subplot(2, 1, 2)
mp.title('Hour Feature Importance', fontsize=16)
mp.ylabel('Feature Importance', fontsize=14)
x = np.arange(hour_fi.size)
# 对特征重要性进行排序，得到有序索引
sorted_indices = hour_fi.argsort()[::-1]  # 倒排
hour_fi = hour_fi[sorted_indices]
mp.grid(linestyle=':')
# 对x轴刻度进行处理
mp.xticks(x, hour_header[sorted_indices])
mp.bar(x, hour_fi, 0.8, color='orangered', label='Day Feature Importance')
mp.legend()
mp.tight_layout()

mp.show()
