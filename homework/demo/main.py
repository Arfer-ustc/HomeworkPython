# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 下午4:35
# @Author  : xuef
# @FileName: main.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_42118777/article
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import svm


data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

print(data_train.describe())
print(data_train.info())


#   subplot2grid(shape,loc)
#让子区跨越固定的网格布局
#subplot2grid 可以自定义排版，使图片更紧凑，第二个参数第几行的第几个图
plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# plots a bar graph of those who surived vs those who did not.
plt.title("Survived") # puts a title on our graph
plt.ylabel("num")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("num")
plt.title("Ticket class")

#已散步图形式展示不同生存性下的年龄分布
plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("age")                         # sets the y axis lable
plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
plt.title("Age in years")

#曲线图
plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("age")# plots an axis lable
plt.ylabel("density")
plt.title("Age distribution of passengers of all levels")
plt.legend(('1st', '2nd', '3rd'),loc='best') # sets our legend for our graph.

#柱状图
plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("Port of Embarkation")
plt.ylabel("num")

plt.show()

# 各变量的相关系数
# Pandas对象对数据进行数据相关性分析，离散数据连续化等：
# 相关性分析
#
print(data_train["Survived"].corr(data_train['Age']))
print(data_train["Survived"].corr(data_train['Pclass']))
data_train[data_train['Sex'] == 'male'] = 0
data_train[data_train['Sex'] == 'female'] = 1
print(data_train["Survived"].corr(data_train['Sex']))
print(data_train["Survived"].corr(data_train['SibSp']))
print(data_train["Survived"].corr(data_train['Parch']))
print(data_train["Survived"].corr(data_train['Embarked']))
print(data_train.corr())

'''
不同舱位/乘客等级可能和财富/地位有关系，最后获救概率可能会不一样
年龄对获救概率也一定是有影响的
和登船港口是不是有关系呢？也许登船港口不同，人的出身地位不同？
'''

# 检验Survived与Pclass的关系
survived0_pclass = data_train[data_train["Survived"] == 0]["Pclass"].value_counts()
survived1_pclass = data_train[data_train["Survived"] == 1]["Pclass"].value_counts()
# print(survived0_pclass)
fig, axes = plt.subplots(2,1)
survived0_pclass.plot(kind='bar', rot = 0, ax = axes[0], title = '0-pclass')
survived1_pclass.plot(kind='bar', rot = 0, ax = axes[1], title = '1-pclass')
plt.show()

# 检验survived与年龄的关系
survived0_age = data_train[data_train["Survived"] == 0]["Age"].value_counts().sort_index()
survived1_age = data_train[data_train["Survived"] == 1]["Age"].value_counts().sort_index()
# print(survived0_age)
fig, axes = plt.subplots(1,1) # 画在同一个subplot中
survived0_age.plot(ax = axes, title = '0-age')
survived1_age.plot(ax = axes, title = '1-age')
plt.legend(('0-age', '1-age'),loc='best')
plt.show()

# 检验Survival与性别的关系
survived0_sex = data_train[data_train["Survived"] == 0]["Sex"].value_counts()
survived1_sex = data_train[data_train["Survived"] == 1]["Sex"].value_counts()
# print(survived1_sex)
fig, axes = plt.subplots(2,1)
survived0_sex.plot(kind='bar', rot = 0, ax = axes[0], title = '0-Sex')
survived1_sex.plot(kind='bar', rot = 0, ax = axes[1], title = '1-Sex')
plt.show()

# 检验Survived与SibSp的关系  船上兄弟姐们或配偶的数量
# survived0_sibsp = data_train[data_train["Survived"] == 0]["SibSp"].value_counts().sort_index()
# survived1_sibsp = data_train[data_train["Survived"] == 1]["SibSp"].value_counts().sort_index()
# # print(survived1_sibsp)
# fig, axes = plt.subplots(2,1)
# survived0_sibsp.plot(kind='bar', rot = 0, ax = axes[0], title = '0-sibsp')
# survived1_sibsp.plot(kind='bar', rot = 0, ax = axes[1], title = '1-sibsp')
# plt.show()
# group_sinsp_survived = data_train.groupby(by=["SibSp", "Survived"])
# print(type(group_sinsp_survived.count()), group_sinsp_survived.count()["PassengerId"])


# 检验Survived与Parch的关系  船上父母数量
# survived0_parch = data_train[data_train["Survived"] == 0]["Parch"].value_counts().sort_index()
# survived1_parch = data_train[data_train["Survived"] == 1]["Parch"].value_counts().sort_index()
# # print(survived1_parch)
# fig, axes = plt.subplots(2,1)
# survived0_parch.plot(kind='bar', rot = 0, ax = axes[0], title = '0-Parch')
# survived1_parch.plot(kind='bar', rot = 0, ax = axes[1], title = '1-Parch')
# plt.show()
# group_parch_survived = data_train.groupby(by=["Parch", "Survived"])
# print(type(group_parch_survived.count()), group_parch_survived.count()["PassengerId"])

# 检验存Survival与Embarked的关系  登船地点
# survived0_embarked = data_train[data_train["Survived"] == 0]["Embarked"].value_counts().sort_index()
# survived1_embarked = data_train[data_train["Survived"] == 1]["Embarked"].value_counts().sort_index()
# # print(survived1_embarked)
# fig, axes = plt.subplots(2,1)
# survived0_embarked.plot(kind='bar', rot = 0, ax = axes[0], title = '0-Embarked')
# survived1_embarked.plot(kind='bar', rot = 0, ax = axes[1], title = '1-Embarked')
# plt.show()

# 计算登船港口和乘客等级的关系
pclass1_embarked = data_train[data_train["Pclass"] == 1]["Embarked"].value_counts().sort_index()
pclass2_embarked = data_train[data_train["Pclass"] == 2]["Embarked"].value_counts().sort_index()
pclass3_embarked = data_train[data_train["Pclass"] == 3]["Embarked"].value_counts().sort_index()
# print(pclass1_embarked)
fig, axes = plt.subplots(3,1)
pclass1_embarked.plot(kind='bar', rot = 0, ax = axes[0], title = 'pclass1_embarked')
pclass2_embarked.plot(kind='bar', rot = 0, ax = axes[1], title = 'pclass2_embarked')
pclass3_embarked.plot(kind='bar', rot = 0, ax = axes[2], title = 'pclass3_embarked')
plt.show()


# 计算不同的年龄不同的pclass所对应的存活情况
group_pclass_age_survived = data_train.groupby(by=["Pclass", "Age", "Survived"])
# print(type(group_pclass_age_survived.count()), group_pclass_age_survived.count()["PassengerId"])

fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度
plt.title(u"根据舱等级和性别的获救情况")

ax1=fig.add_subplot(2, 3, 1)
# print(data_train[(data_train["Sex"]=="female") & (data_train["Pclass"] ==1)]["Survived"].value_counts())
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 1].value_counts().plot(kind='bar', label="female pclass1", color='#FA2479')
ax1.set_xticklabels(["rescued", "unrescued"], rotation=0)
ax1.legend(["female/pclass1"], loc='best')

ax2=fig.add_subplot(2, 3, 2, sharey = ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 2].value_counts().plot(kind='bar', label="female pclass2", color='#FA2479')
ax2.set_xticklabels(["rescued", "unrescued"], rotation=0)
ax2.legend(["female/pclass2"], loc='best')

ax3=fig.add_subplot(2, 3, 3, sharey = ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label="female pclass3", color='#FA2479')
ax3.set_xticklabels(["rescued", "unrescued"], rotation=0)
ax3.legend(["female/pclass3"], loc='best')

ax4=fig.add_subplot(2, 3, 4, sharey = ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 1].value_counts().plot(kind='bar', label="male pclass1", color='#FA2479')
ax4.set_xticklabels(["rescued", "unrescued"], rotation=0)
ax4.legend(["male/pclass1"], loc='best')

ax5=fig.add_subplot(2, 3, 5, sharey = ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 2].value_counts().plot(kind='bar', label="male pclass2", color='#FA2479')
ax5.set_xticklabels(["rescued", "unrescued"], rotation=0)
ax5.legend(["male/pclass2"], loc='best')

ax6=fig.add_subplot(2, 3, 6, sharey = ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label="male pclass3", color='#FA2479')
ax6.set_xticklabels(["rescued", "unrescued"], rotation=0)
ax6.legend(["male/pclass3"], loc='best')

plt.show()


# print(data_train["Cabin"].unique(), data_train["Cabin"].unique().size)
# 148类
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({'notnull':Survived_cabin, 'isnull':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title("Cabin-Survived")
plt.xlabel("Cabin-is/not null")
plt.ylabel("num")
plt.show()

# print(data_train["Age"].isnull().value_counts())
# Age: 177缺失值

# 采用随机森林算法填充缺失数据
def set_null_age(data):
    df = data[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    age_notnull = df[df["Age"].notnull()].as_matrix()
    age_null = df[df["Age"].isnull()].as_matrix()
    '''
    RandomForest是一个用在原始数据中做不同采样，
    建立多颗DecisionTree，再进行average等等来降低过拟合现象，提高结果的机器学习算法
    '''

    rfr = RandomForestRegressor(n_estimators=10000, n_jobs=-1, random_state=0)
    rfr.fit(age_notnull[:,1:], age_notnull[:,0])

    pre_age = rfr.predict(age_null[:,1:])

    data.loc[data["Age"].isnull(), "Age"] = pre_age

    return  data, rfr

print(data_train["Age"].shape)
# print(data_train["Cabin"].isnull().value_counts())
# Cabin：687 na

def set_null_cabin(data):
    data.loc[data["Cabin"].notnull(), "Cabin"] = "notnull"
    data.loc[data["Cabin"].isnull(), "Cabin"] = "isnull"
    # 还可采用np.where函数
    # np.where(data_train["Cabin"].isnull(), "isnull", "notnull")
    return data

data_train, rfr = set_null_age(data_train)
data_train = set_null_cabin(data_train)

# 将离散值（类型值）改为数值

dummies_sex = pd.get_dummies(data_train["Sex"], prefix = "Sex")
dummies_cabin = pd.get_dummies(data_train["Cabin"], prefix = "Cabin")
dummies_embarked = pd.get_dummies(data_train["Embarked"], prefix = "Embarked")

# print(dummies_sex.columns)

data_train = pd.concat([data_train, dummies_sex, dummies_cabin, dummies_embarked], axis=1)
data_train.drop(["Sex", "Cabin", "Ticket", "Embarked"], axis=1, inplace=True)

# print(data_train.head(6))

# 数据标准化 Age、Fare 且保存训练集上均值和标准差从而在测试集同样处理
scaler = preprocessing.StandardScaler()
# age_fit_parmer = scaler.fit(np.ndarray(data_train["Age"]))
# data_train["Age_scale"] = scaler.fit_transform(data_train["Age"], age_fit_parmer)
# # data_train["Age"].hasnull()
# fare_fit_parmer = scaler.fit(np.ndarray(data_train["Fare"]))
# data_train["Fare_scale"] = scaler.fit_transform(data_train["Fare"], fare_fit_parmer)

#feature scalling age fare 差距过大 不利于模型收敛
age_fare_fit_parmer = scaler.fit(data_train[["Age", "Fare"]])
# data_train[["Age_scale", "Fare_scale"]] = \
age_fare_scale = scaler.fit_transform(data_train[["Age", "Fare"]], age_fare_fit_parmer)
data_train["Age_scale"] = age_fare_scale[:,0]
data_train["Fare_scale"] = age_fare_scale[:,1]
# print(type(scaler.fit_transform(data_train[["Age", "Fare"]], age_fare_fit_parmer)))
# <class 'numpy.ndarray'> (890, 2)
# data_train["Age"].hasnull()


# 采用LogisticRegression算法建模
train_df = data_train.filter(regex="Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass")
print(train_df.head())
train_np = train_df.as_matrix()

#特征属性值
train_X = train_np[:, 1:]
#survival结果
train_y = train_np[:, 0]
print(train_X.shape)
clf = LogisticRegression(C=1, penalty='l1', tol=1e-6, solver='liblinear')
clf.fit(train_X, train_y)

#对test进行预处理

data_test = pd.read_csv("test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_null_cabin(data_test)
dummies_cabin_test = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_embarked_test = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_sex_test = pd.get_dummies(data_test['Sex'], prefix= 'Sex')


data_test = pd.concat([data_test, dummies_cabin_test, dummies_embarked_test, dummies_sex_test], axis=1)
data_test.drop(['Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

age_fare_scale2 = scaler.fit_transform(data_test[["Age", "Fare"]], age_fare_fit_parmer)
data_test["Age_scale"] = age_fare_scale2[:,0]
data_test["Fare_scale"] = age_fare_scale2[:,1]
# data_test['Age_scaled'] = scaler.fit_transform(data_test['Age'], age_fit_parmer)
# data_test['Fare_scaled'] = scaler.fit_transform(data_test['Fare'], fare_fit_parmer)

# 采用LogisticRegression算法
test_df = data_test.filter(regex="Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass")
print(test_df.head())
test_np = test_df.as_matrix()
print(test_np.shape)
predictions = clf.predict(test_np)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv", index=False)

result_test = pd.read_csv("logistic_regression_predictions.csv")
print(result_test["Survived"].value_counts())


# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.scatter(data_train[""], train_y, color='black', zorder=20)
# X_test = np.linspace(-5, 10, 300)
#
# def model(x):
#     return 1 / (1 + np.exp(-x))
#
# loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
# plt.plot(X_test, loss, color='red', linewidth=3)


#绘制学习曲线，以确定模型的状况
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs = 1,
                        train_sizes=np.linspace(.05, 1., 20)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)#求均值
    train_scores_std = np.std(train_scores, axis=1)#方差计算
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

#少样本的情况情况下绘出学习曲线
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
midpoint, diff = plot_learning_curve(clf, "LR-learning_curve",
                    train_np[:,1:], train_np[:,0], ylim=(0.5, 1.01), cv=cv,
                    train_sizes=np.linspace(.05, 0.2, 5))
print(midpoint, diff)

# 采用svm分类器
svm_clf = svm.SVC(C=1, kernel="rbf", gamma="auto", tol=1e-6, )
pre_svm = svm_clf.fit(train_X, train_y)

predictions_svm = svm_clf.predict(test_np)
result_svm = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions_svm.astype(np.int32)})
result_svm.to_csv("logistic_regression_predictions2.csv", index=False)

midpoint_svm, diff_svm = plot_learning_curve(svm_clf, "SVM-learning_curve",
                    train_np[:,1:], train_np[:,0], ylim=(0.5, 1.01), cv=cv,
                    train_sizes=np.linspace(.05, 0.2, 5))
#numpy.linspace 在指定间隔内 返回指定个数的 数
print(midpoint_svm, diff_svm)
# 0.8103784010046847 0.06989222461055977