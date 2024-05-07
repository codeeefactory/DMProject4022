import matplotlib
import numpy as np
from sklearn import tree, __all__
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.naive_bayes import GaussianNB,CategoricalNB,ComplementNB,BernoulliNB,MultinomialNB
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot  as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb
from ydata_profiling import ProfileReport
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
# fetch dataset

breast_cancer = fetch_ucirepo(id=14)
df=pd.DataFrame.from_dict(breast_cancer.data.features)
# print(df)
# print(df.describe())
# print(df.columns)
# print(df.info())
# print(df.dtypes)
# data (as pandas dataframes)
X = breast_cancer.data.features
X=X.iloc[:,:-1]
Y = breast_cancer.data.labels
df["irradiat"]=df["irradiat"].apply(lambda x:0 if x=="no" else 1)
Y = df.iloc[:,-1]

profile=ProfileReport(df)
profile.to_file("your_report.html")
driver.get("")
scaler=StandardScaler()
g=sb.FacetGrid(df)
g.map(sb.distplot)
# print(df["irradiat"].unique())

# labels,mapping=pd.factorize(df["irradiat"].unique())[0]
# df["irradiat"]=pd.factorize(df["irradiat"])[0]
# encoder=LabelEncoder()
# #scaler_data=scaler.fit_transform(df)
# col=df.columns
# for c in col:
#     df[c]=encoder.fit_transform(df[c])
# mmscaler=MinMaxScaler()
# df2=mmscaler.fit_transform(df)
# scaler_data2=scaler.fit_transform(df)
#
# # plt.figure(figsize=(100,50))
# # plt.xticks(fontsize=72)
# # plt.yticks(fontsize=60)
# # plt.xlabel("features",fontsize=20)
# # plt.ylabel("faravani",fontsize=20)
# # sb.boxplot(data=df)
# # plt.plot()
# # plt.savefig("fig.jpg")
# # print(labels,mapping)
# # print(df.nunique().tolist())
# # y = breast_cancer.data.targets
# # z = breast_cancer.data
#
# # print(type(z))
# # print(z)
# # metadata
# # print(breast_cancer.metadata)
# #
# # # variable information
# # print(breast_cancer.variables)
# var_list=[]
# dropna=df.isnull()/len(df)*100
# variance=df.var()
# print(type(variance))
# print(variance)
# print(dropna)
# # for i in range(len(variance)):
# #     if variance.iloc[i]>=0.10:
# #         var_list.append(str(variance.index[i]))



# rfr=RandomForestRegressor(random_state=1,max_depth=10)
# dataset=pd.get_dummies(df.iloc[:,:-1])
# #
# dum=rfr.fit(dataset,df.iloc[:,-1])
# feature=dataset.columns
# importance=rfr.feature_importances_
# print(dum)
#
# indices = np.argsort(importance)[-9:]  # top 10 features
# featureimp=[feature[i] for i in indices]
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importance[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [feature[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.plot()
# plt.savefig("feature.jpg")
# plt.show()
# print(var_list)
# print(set(cor_list))
LE=LabelEncoder()
for i in X.columns:
    X[i]=LE.fit_transform(X[i].astype(str))
cor=X.corr()
cor_list=[]
print()
print(cor)
for i in range(cor.shape[0]):
    for j in range(cor.shape[1]):
        if (i!=j)and cor.iloc[i,j]>=0.10:
            cor_list.append(cor.index[i])
            break
X=X[cor_list]


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.7
)
dtc=DecisionTreeClassifier(random_state=0,max_depth=4)
dtc=dtc.fit(X_train,y_train)
tree.plot_tree(dtc,fontsize=14)
print()
plt.plot()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(120, 60)
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
y_predict=rfc.predict(X_test)
print("accuracy of the model:",metrics.accuracy_score(y_test,y_predict))

fig.savefig('treeeeeeeeee5321555456114645665466446514e1.jpg')
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_predict_gnb=gnb.predict(X_test)
print("accuracy of the model(GNB):",metrics.accuracy_score(y_test,y_predict_gnb))
bnb=BernoulliNB()
bnb.fit(X_train,y_train)
y_predict_bnb=bnb.predict(X_test)
print("accuracy of the model(GNB):",metrics.accuracy_score(y_test,y_predict_bnb))
cnb=ComplementNB()
cnb.fit(X_train,y_train)
y_predict_cnb=cnb.predict(X_test)
print("accuracy of the model(GNB):",metrics.accuracy_score(y_test,y_predict_cnb))
mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_predict_mnb=mnb.predict(X_test)
print("accuracy of the model(MNB):",metrics.accuracy_score(y_test,y_predict_mnb))
canb=CategoricalNB()
canb.fit(X_train,y_train)
y_predict_canb=canb.predict(X_test)
print("accuracy of the model(CANB):",metrics.accuracy_score(y_test,y_predict_canb))

