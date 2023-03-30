# -*- coding: UTF-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 加载 sklearn 自带的乳腺癌（分类）数据集
X, y = load_breast_cancer(return_X_y=True)

# 以指定比例将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.875, test_size=0.125, random_state=188
)

# 使用 lr 类，初始化模型
clf = LogisticRegression(
    penalty="l2", C=1.0, random_state=None, solver="lbfgs", max_iter=3000,
    multi_class='ovr', verbose=0,
)

# 使用训练数据来学习（拟合），不需要返回值，训练的结果都在对象内部变量中
clf.fit(X_train, y_train)

# 使用测试数据来预测，返回值预测分类数据
y_pred = clf.predict(X_test)

# 打印主要分类指标的文本报告
print('--- report ---')
print(classification_report(y_test, y_pred))

# 打印模型的参数
print('--- params ---')
print(clf.coef_, clf.intercept_)

# 打印 auc
print('--- auc ---')
print(roc_auc_score(y_test, y_pred))
