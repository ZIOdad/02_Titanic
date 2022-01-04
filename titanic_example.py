# work flow

# 1. 데이터분류
# 2. 연관성 찾기 -> 입력값과 결과값의 연관성을 파악
# 3. 텍스트 정보의 디지털화
# 4. None 값 채우기
# 5. 데이터 시각화
# 6. 모델 생성 및 평가

import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

print(__file__)  # 현재 code파일 위치
print(os.getcwd())  # current working directory
print(os.path.dirname(os.path.realpath(__file__)))

os.chdir(os.path.dirname(os.path.realpath(__file__)))  # working directory 변경
print(os.getcwd())

warnings.filterwarnings(action='ignore')

#dataframe 출력 옵션
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
#pd.describe_option() #옵션설명

"""1. 데이터 불러오기 (데이터를 정제하기 위한 툴)"""

# 시각화 모듈

# 머신러닝 모듈

# 데이터셋 불러오기
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
# print(train_df.column)
# 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
# 승객번호, 생존여부, 객실등급, 이름, 성별, 나이, 동반한 형제자매와 배우자수, 동반한 부모/자녀수, 티켓번호, 요금, 객실번호, 승선항

print('-' * 100)
print(train_df.info())
print('-' * 100)
print(test_df.info())
print('-' * 100)
print(train_df.isnull().sum())
#비어있는 개수: 나이 177개, 객실번호 687개, 승선항 2개

print(train_df.describe())
#훈련데이터의 통계분석(object type의 데이터는 불가)
"""체크포인트
1. 총 2,224명의 승객 중 훈련데이터는 891개임. -> 약 40% 정도
2. 살았는지 죽었는지 여부는 0과 1로 구분
3. 타이타닉 사고의 실제 생존률 32% -> 훈련데이터셋의 생존률은 38.4%
4. Parch 항목의 결과, 75%이상의 승객이 혼자 탑승함.
5. SibSp 항목의 결과, 상위 25% 내 승객은 형제자매/배우자와 함께 탑승함.
6. Age 항목의 결과, 평균 29.7세이므로 대부분 젋은 승객들이나, 최대 80세 노인도 탑승함.
7. Pclass 항목의 결과, 3등급 객실에 탑승한승객이 가장 많음.
"""
print('-' * 100)
print(train_df.describe(include=['O'])) #문자열(string) include=['O']
#훈련데이터 object 변수 통계분석
"""체크포인트
1. 이름항목의 unique값이 891이므로 동명이인은 없음.
2. 남성(male)의 비율이 64%(577/891)임.
3. Cabin 항목의 결과, 다인실도 존재함을 알 수 있음.
4. Embarked 항목의 결과, 승선항 S에서 탑승한 승객이 가장 많음.
"""
print('-' * 100)
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#groupby에 as_index를 False로 하면 Pclass를 index로 사용하지 않음
# ascending : 오름차순
# as_index를 True로 하면 Pclass를 index로 사용
"""체크포인트
1. 객실등급이 높을수록 생존율이 높음.
"""
print('-' * 100)
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
"""체크포인트
1. 여성의 생존율이 남성보다 높음.
"""
print('-' * 100)
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('-' * 100)
print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
"""체크포인트
1. 동행자 수가 작을수록 생존율이 높음.
"""
plt.hist(train_df['Age'], bins=20)
grid = sns.FacetGrid(train_df, col='Survived') # 열(col)을 생존 여부로 나눔
grid.map(plt.hist, 'Age', bins=20) # 히스토그램 시각화, 연령 별 분포, 히스토그램 bin을 20개로 설정
"""체크포인트
1. 4세 이하 유아가 생존율이 높음.
2. 많은 탑승객이 분포된 20~35세에서 생존자가 많음.
"""
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', hue='Pclass', height=2.2, aspect=1.6) # width = height * aspect
grid.map(plt.hist, 'Age', alpha=0.5, bins=20) # 투영도(alpha): 0.5
grid.add_legend()
"""체크포인트
1. 3등급 탑승객의 생존율이 가장 낮음.
2. 2등급 탑승객 중 유아들은 대부분 생존함.
3. 1등급 탑승객의 생존율이 상대적으로 높음.
"""
grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order = [1, 2, 3], hue_order = ["male", "female"])
# Pointplot 시각화, x: 객실 등급, y: 생존 여부, 색깔: 성별, x축 순서: [1, 2, 3], 색깔 순서: [남성, 여성]
grid.add_legend()

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None,order=["male","female"])
# 바그래프로 시각화, x: 성별, y: 요금, Error bar: 표시 안 함
grid.add_legend()
#plt.show()


"""2. 데이터 전처리"""
print('-' * 100)
print("Before", train_df.shape, test_df.shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1) # 열(axis=1)제거
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1) # 열(axis=1)제거
combine = [train_df, test_df]
print("After", train_df.shape, test_df.shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#dataset이라는 df에 Title(column)을 추가하는데, Name column에서 대문자로 시작해서 소문자로 나열, .을 만나면 탐색을 멈추고 string을 추출
print('-' * 100)
print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
#Miss(Mlle, Ms), Mrs(Mme), Master, Mr 를 제외한 나머지는 Rare로 분류
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print('-' * 100)
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
#title 변수를 숫자형으로 바꿔줌.
print('-' * 100)
print(train_df.head(5))
print('-' * 100)
print(test_df.head(5))

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
#필요없는 데이터 삭제
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#성별을 숫자형 데이터로 변경
print(train_df.shape, test_df.shape)
print('-' * 100)
print(train_df.head(5))
print('-' * 100)
print(test_df.head(5))

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()
            # 위에서 guess_ages사이즈를 [2,3]으로 잡아뒀으므로 j의 범위도 이를 따름

            age_guess = guess_df.median()

            # age의 random값의 소수점을 .5에 가깝도록 변형
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

print('-' * 100)
print(train_df.isnull().sum())

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
# 임의로 5개 그룹을 지정
print('-' * 100)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print('-' * 100)
print(train_df.head())

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print('-' * 100)
print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print('-' * 100)
print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
print('-' * 100)
print(train_df.head())

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
print('-' * 100)
print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

freq_port = train_df.Embarked.dropna().mode()[0]
print('-' * 100)
print(freq_port)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
print('-' * 100)
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
print('-' * 100)
print(train_df.head())

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print('-' * 100)
print(test_df.head())

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print('-' * 100)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
print('-' * 100)
print(train_df.head(10))
print('-' * 100)
print(test_df.head(10))

"""3. 데이터 준비"""
# 목적 변수 제거
X_train = train_df.drop("Survived", axis=1)
#목적 변수 역할
Y_train = train_df["Survived"]
#예측 대상 데이터 셋
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

"""4. Logistic Regression"""
print('-' * 100)
print('-' * 100)
print('-------------Training results--------------')
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print('-' * 100)
print('Accuracy of Logisitic Regression :', acc_log)

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
print('-' * 100)
print(coeff_df.sort_values(by='Correlation', ascending=False))

"""5. SVC(Support Vector Machines)"""
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print('-' * 100)
print('Accuracy of SVC :', acc_svc)

"""6. K-NN(K Nearest Neighberhood)"""
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print('-' * 100)
print('Accuracy of K-NN :', acc_knn)

"""7. Gaussian Naive Bayes"""
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print('-' * 100)
print('Accuracy of Gaussian Naive Bayes :', acc_gaussian)

"""8. Perceptron"""
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print('-' * 100)
print('Accuracy of Perceptron :', acc_perceptron)

"""9. Linear SVC"""
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print('-' * 100)
print('Accuracy of Linear SVC :', acc_linear_svc)

"""10. SGD(Stochastic Gradient Descent)"""
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print('-' * 100)
print('Accuracy of SGD :', acc_sgd)

"""11. Decision Tree"""
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('-' * 100)
print('Accuracy of Decision Tree :', acc_decision_tree)

"""12. Random Forest"""
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('-' * 100)
print('Accuracy of Random Forest :', acc_random_forest)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print('-' * 100)
print(models.sort_values(by='Score', ascending=False))

#Random Forest 모델 결과
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
print('-' * 100)
print(submission)
