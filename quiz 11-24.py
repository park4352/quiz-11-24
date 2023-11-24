import pandas as pd
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# 산점도 그래프 출력을 위한 pandas의 Sca
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터 파일 경로
filename = "./09_irisdata.csv"

# 컬럼 이름 정의
column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# 데이터 읽어오기
df = pd.read_csv(filename, names=column_names)

# 데이터 독립 변수(X)와 종속 변수(y)로 나누기
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


from pandas.plotting import scatter_matrix

scatter_matrix(df)
plt.savefig("./09_irisdata.png")


model = DecisionTreeClassifier(max_depth=1000, min_samples_split=60, min_samples_leaf=5)

kfold = KFold(n_splits=10, random_state=5, shuffle=True)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, y, cv=kfold)
# 결과 출력
print(results.mean())
