# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %% [markdown]
# # 1st look at the data

# %%
df = pd.read_excel('Assignment-2024-training-data-set.xlsx')

# %%
df.head()

# %%
df.info()

# %%
print(df.shape[0])
print(df.shape[1])

# %% [markdown]
# # Irrelevant attributes

# %% [markdown]
# ## rate
# no info on the data

# %%
plt.hist(df.rate);
# have no idea

# %%
plt.scatter(df.rate,df.label);

# %% [markdown]
# ## protocol

# %%
df.proto.unique()
# i just dont think it useful

# %% [markdown]
# ## service

# %%
df.service.unique()

# %% [markdown]
# ## state

# %%
df.state.unique()

# %% [markdown]
# ## attack_cat: normal  = 1 ?

# %%
df.attack_cat.unique()

# %%
print(df.label.loc[df.attack_cat == "Normal"].unique())
#yep
print(df.label.loc[df.attack_cat != "Normal"].unique())

# %% [markdown]
# # Missing entries

# %%
df.isnull().sum()  #couldn't find from function

# %%
#check unique value
for i in range(len(df.columns)):
    print(df.columns[i])
    print(pd.unique(df.iloc[:,i]))  

# %% [markdown]
# ## duration = 0 ?
# i dont think duration of time saction should be zero

# %%
# check how much are there
df.dur[df.dur == 0].count()
#small amount

# %%
temp = ["dur","sbytes","dbytes"]
df[temp][df.dur == 0][(df.sbytes != 0) | (df.dbytes != 0)].count()

# %%
# decide to replace with mean
df.dur =  df.dur.replace(0,df.dur.mean())
df.dur[df.dur == 0].count()

# %% [markdown]
# ## rate 
# the value shouldn't be zero
# going to remove anyway

# %%
df.rate[df.rate == 0].count()

# %% [markdown]
# ## sbytes
# shouldn't be zero unless the thier no data transfer from the  source to destination

# %%
df.sbytes[df.sbytes == 0].count()

# %% [markdown]
# ##  dbytes
# shouldn't be zero unless the thier no data transfer from the  destination to source

# %%
df.dbytes[df.dbytes == 0].count()


# %% [markdown]
# ## sload and dload
# bps should be more than 0 except no data transfer

# %%
df.sload[(df.sload == 0) & (df.sbytes != 0)].count()
# Since their always a data transfer from the variable sbyte the value should not be zero
# low amount of data missing replace with mean

# %%
df.sload =  df.sload.replace(0,df.sload.mean())
df.sload[df.sload == 0].count()

# %%
df.dload[df.dload == 0].count()

# %%
#there should be 7 missing value
df.dload[(df.dload == 0) & (df.dbytes != 0)].count()

# %%
#replace them with mean
df.dload[(df.dload == 0) & (df.dbytes != 0)] = df.dload[(df.dload == 0) & (df.dbytes != 0)].replace(0,df.dload.mean())
df.dload[df.dload == 0].count() 

# %% [markdown]
# ## swin and dwin have low unique value?
# apon reseach the number could be zero.

# %%
print(df.swin.unique())
plt.hist(df.swin);

# %%
print(df.dwin.unique())
plt.hist(df.dwin);

# %% [markdown]
# ## sjit and djit

# %%
print(df.sjit.loc[df.sjit == 0].count())
print(df.sjit.loc[df.sjit != 0].count())
# apon research the data can be zero

# %%
print(df.djit.loc[df.djit == 0].count())
print(df.djit.loc[df.djit != 0].count())
# apon research the data can be zero

# %% [markdown]
# # duplicate row

# %%
print(df.shape[0])
print(df.duplicated(keep=False).sum()) #there are lots of duplicate data

# %%
df.loc[(df.duplicated(keep=False))].sort_values("dur")

# %%
# cal the ratio
df.duplicated().sum()/df.shape[0]
# 38  percent duplicate data
# since it is the log data of im going to keep all the duplicate

# %% [markdown]
# # duplicate column

# %% [markdown]
# ## sload and sbyte Dload and dbyte

# %%
plt.scatter(df.sload*df.dur, df.sbytes);

# %%
plt.scatter(df.dload*df.dur, df.dbytes);

# %% [markdown]
# ## sttl and dttl

# %%
df.dttl.head()

# %%
df.sttl.head()
# no they not the same

# %% [markdown]
# ## synack and sckdat and tcprtt

# %%
df.synack.head()

# %%
df.ackdat.head()

# %%
plt.scatter(df.synack,df.ackdat);
# look like they are not the same data

# %%
temp = df.synack + df.ackdat
plt.scatter(temp, df.tcprtt);
#yep they are the same data
# remove 

# %% [markdown]
# ## is_ftp_login and ct_ftp_cmd
# the feature is ftp login should be binary but have the same unique value as ct_ftp_cmd

# %%
pd.unique(df.is_ftp_login)

# %%
# check if that true
plt.scatter(df.is_ftp_login, df.ct_ftp_cmd);
#look like it 
df.is_ftp_login.equals(df.ct_ftp_cmd)
# this comfirm it
# is_ftp_login should be drop since from the feature description the data should hold binary

# %% [markdown]
# # Data type

# %% [markdown]
# ## sbytes and dbytes
# do every file have to have a full byte?
# the ans is yes wow

# %% [markdown]
# # Feature/Attribute selection

# %% [markdown]
# remove tcprtt = df.synack + df.ackdat
# Sload > we have duration and sbyte
# Dload > we have duration and dbyte
# attack_cat

# %%
drop_column = ["tcprtt","sload", "dload","attack_cat","is_ftp_login"]
df = df.drop(columns= drop_column, errors ='ignore')

# %% [markdown]
# #  Scaling and standardisation

# %%
#check unique value
for i in range(len(df.columns)):
    print(df.columns[i], df.iloc[:,i].min(), df.iloc[:,i].max())


# %%
from sklearn.preprocessing import MinMaxScaler
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns
scaler = MinMaxScaler()
scaler = scaler.fit(df.loc[:, ~df.columns.isin(categorical_columns)])
df.loc[:, ~df.columns.isin(categorical_columns)]= scaler.transform(df.loc[:, ~df.columns.isin(categorical_columns)])

# %%
#check unique value
for i in range(len(df.columns)):
    print(df.columns[i], df.iloc[:,i].min(), df.iloc[:,i].max())

# %% [markdown]
# #  Data imbalance

# %%
# check the amount of normal and attack data
print(df.label.loc[df.label == 0].count())
print(df.label.loc[df.label == 1].count())
# there are data imbalance in the data 

# %%
print(df.label.loc[df.label == 1].count()/df.shape[0])
#the attack data account for 73 percent of the dataset

# %%
from sklearn.utils import resample
df_major = df[df.label == 1]
df_minor = df[df.label == 0]

df_major_downs = resample(df_major, replace = True, n_samples = df_minor.shape[0], random_state = 5601)
df = pd.concat([df_minor, df_major_downs])
df.reset_index(inplace=True, drop=True)

# %% [markdown]
# # Feature engineering

# %%
from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(sparse_output=False, handle_unknown= 'ignore')
encoder = encoder.fit(df[categorical_columns])
one_hot_encoded = encoder.transform(df[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded = pd.concat([df, one_hot_df], axis=1 )
df_encoded = df_encoded.drop(categorical_columns, axis=1)
df_encoded.info()


# %% [markdown]
# # Training, Validation, and Test Sets

# %%
y = df_encoded.label
X = df_encoded.loc[:,df_encoded.columns != "label"]
X.info()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5601)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=5601) 

# %%
print(X_train.info())
print(X_val.info())
print(X_test.info())

# %%
# Save the data to arff
import arff
#training
#join file back
train_df = pd.concat([X_train,y_train],axis = 1)
print(train_df.info())
arff.dump('Training.arff'
      , train_df.values
      ,relation='relation name'
      , names=train_df.columns)

#validation
val_df = pd.concat([X_val,y_val],axis = 1)
print(val_df.info())
arff.dump('Validation.arff'
      , val_df.values
      ,relation='relation name'
      , names=val_df.columns)

#testing
test_df = pd.concat([X_test,y_test],axis = 1)
print(test_df.info())
arff.dump('Test.arff'
      , test_df.values
      ,relation='relation name'
      , names=test_df.columns)


# %% [markdown]
# #  k-NN

# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# %%
y_pred_knn = knn.predict(X_test)

# %%
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# %% [markdown]
# #  Naive Bayes

# %%
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# %%
y_pred_nb = nb.predict(X_test)

# %%
confusion_matrix(y_test, y_pred_nb)

# %%
print(classification_report(y_test, y_pred_nb))

# %% [markdown]
# #  Decision Trees

# %%
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=5601)
tree = tree.fit(X_train, y_train)

# %%
y_pred_tree = tree.predict(X_test)

# %%
confusion_matrix(y_test, y_pred_tree)

# %%
print(classification_report(y_test, y_pred_tree))

# %% [markdown]
# # Cross validation

# %%
from sklearn.model_selection import cross_val_score

# %% [markdown]
# ## knn

# %%
n_neighbors_list = [3, 5, 10, 20, 50,100,200,500,1000]
knns_cv_result = []

for i in n_neighbors_list:
    
    model = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(model, X_val,y_val,cv = 5)
    knns_cv_result.append({
        "n_neighbors": i,
        "Mean CV Score": scores.mean(),
        "Std CV Score": scores.std()
    })
score_knns_cv = pd.DataFrame(knns_cv_result)

# %%
score_knns_cv

# %% [markdown]
# ## Naive Bayes

# %%
nb_cv = GaussianNB()
scores_nb_cv= cross_val_score(nb_cv, X_val,y_val,cv = 5)
scores_nb_cv.mean()

# %% [markdown]
# ## Desiction Tree

# %%
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [1,2,5,10,20,50,100,200,500,1000],
    'min_samples_leaf': [1,2,5,10,20,50,100,200,500,1000],
    'min_samples_split': [2,5,10,20,50,100,200,500,1000],
}
tree = DecisionTreeClassifier(random_state=5601)
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, 
                           cv=5, verbose=True)
grid_search.fit(X_val, y_val)

# Best score and estimator
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)

# %% [markdown]
# ## compare all of the best 3 models

# %%
result_cv = []
knn5 =KNeighborsClassifier(n_neighbors= 5)
score_knn5_cv = cross_val_score(knn5, X_val,y_val,cv = 5)
result_cv.append({
    "Model": "5nn",
    "mean CV score" : score_knn5_cv.mean(),
    "std CV score" : score_knn5_cv.std()
})
result_cv.append({
    "Model": "nb",
    "mean CV score" : scores_nb_cv.mean(),
    "std CV score" : scores_nb_cv.std()
})
btree =DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=5601)
score_btree_cv = cross_val_score(btree, X_val,y_val,cv = 5)
result_cv.append({
    "Model": "btree",
    "mean CV score" : score_btree_cv.mean(),
    "std CV score" : score_btree_cv.std()
})
cv_df = pd.DataFrame(result_cv)
cv_df

# %% [markdown]
# # Classifier comparison

# %%
# confusion martrix
# recreate best models from each knn
knn5 =KNeighborsClassifier(n_neighbors= 5)
knn5 = knn5.fit(X_train,y_train)
y_pred_5nn =  knn5.predict(X_test)
print("5nn")
print(confusion_matrix(y_test, y_pred_5nn))

#nb have only one model
print("nb")
print(confusion_matrix(y_test, y_pred_nb))

# recreate best models from each decision tree
btree =DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=5601)
btree = btree.fit(X_train,y_train)
y_pred_btree =  btree.predict(X_test)
print("btree")
print(confusion_matrix(y_test, y_pred_btree))

# %%
from sklearn.metrics import accuracy_score,precision_score,f1_score
result = []
result.append({
        "Model": "5nn",
        "Accuracy": accuracy_score(y_test,y_pred_5nn),
        "Precision": precision_score(y_test,y_pred_5nn),
        "F1" :f1_score(y_test,y_pred_knn)
        })
result.append({
        "Model": "nb",
        "Accuracy": accuracy_score(y_test,y_pred_nb),
        "Precision": precision_score(y_test,y_pred_nb),
        "F1" :f1_score(y_test,y_pred_nb)
        })
result.append({
        "Model": "tree",
        "Accuracy": accuracy_score(y_test,y_pred_tree),
        "Precision": precision_score(y_test,y_pred_tree),
        "F1" :f1_score(y_test,y_pred_tree)
        })
compair_df = pd.DataFrame(result)
compair_df

# %% [markdown]
# # Test Set import and transformation

# %%
test1  = pd.read_excel('Test-Data-Set-1-2024.xlsx')
test2  = pd.read_excel('Test-Data-Set-2-2024.xlsx')

# %%
test1.info()

# %%
test2.info()

# %%
test1 = test1.drop(columns= drop_column, errors ='ignore')
test1.info()

# %%
#test1 data transformation to the same format as training data
#feature selection
test1 = test1.drop(columns= drop_column, errors ='ignore')
#scaling
test1.loc[:, ~df.columns.isin(categorical_columns)]= scaler.transform(test1.loc[:, ~test1.columns.isin(categorical_columns)])
#one-hot encode
one_hot_encoded_t1 = encoder.transform(test1[categorical_columns])
one_hot_t1 = pd.DataFrame(one_hot_encoded_t1, columns=encoder.get_feature_names_out(categorical_columns))
test1_encoded = pd.concat([test1, one_hot_t1], axis=1)
test1_encoded = test1_encoded.drop(categorical_columns, axis=1)
test1_encoded.info()

# %%
#test2 data transformation to the same format as training data
#feature selection
test2 = test2.drop(columns= drop_column, errors ='ignore')
#one-hot encode
test2.loc[:, ~df.columns.isin(categorical_columns)]= scaler.transform(test2.loc[:, ~test2.columns.isin(categorical_columns)])

one_hot_encoded_t2 = encoder.transform(test2[categorical_columns])
one_hot_t2 = pd.DataFrame(one_hot_encoded_t2, columns=encoder.get_feature_names_out(categorical_columns))
test2_encoded = pd.concat([test2, one_hot_t2], axis=1)
test2_encoded = test2_encoded.drop(categorical_columns, axis=1)
test2_encoded.info()

# %%
y_test1 = test1_encoded.label
X_test1 = test1_encoded.loc[:,test1_encoded.columns != "label"]
y_test2 = test2_encoded.label
X_test2 = test2_encoded.loc[:,test2_encoded.columns != "label"]

# %% [markdown]
# # Test case evaluation

# %% [markdown]
# ## knn

# %%
y_pred1_knn5 = knn5.predict(X_test1)
print(confusion_matrix(y_test1, y_pred1_knn5))
print(classification_report(y_test1, y_pred1_knn5))

# %%
y_pred2_knn5 = knn5.predict(X_test2)
print(confusion_matrix(y_test2, y_pred2_knn5))
print(classification_report(y_test2, y_pred2_knn5))

# %% [markdown]
# ## Naive Bayes

# %%
y_pred1_nb = nb.predict(X_test1)
print(confusion_matrix(y_test1, y_pred1_nb))
print(classification_report(y_test1, y_pred1_nb))

# %%
y_pred2_nb = nb.predict(X_test2)
print(confusion_matrix(y_test2, y_pred2_nb))
print(classification_report(y_test2, y_pred2_nb))

# %% [markdown]
# ## Decision Tress

# %%
y_pred1_btree = btree.predict(X_test1)
print(confusion_matrix(y_test1, y_pred1_btree))
print(classification_report(y_test1, y_pred1_btree))

# %%
y_pred2_btree = btree.predict(X_test2)
print(confusion_matrix(y_test2, y_pred2_btree))
print(classification_report(y_test2, y_pred2_btree))

# %% [markdown]
# ## export csv

# %%
test1_pred = {"ID":range(len(y_pred1_knn5)),
              "Predict1":y_pred1_knn5,
              "Predict2":y_pred1_btree}
test1_pred_df = pd.DataFrame(test1_pred)

test1_pred_df.to_csv('Predict1.csv', index=False)


# %%
test2_pred = {"ID":range(len(y_pred2_knn5)),
              "Predict1":y_pred2_knn5,
              "Predict2":y_pred2_btree}
test2_pred_df = pd.DataFrame(test2_pred)

test2_pred_df.to_csv('Predict2.csv', index=False)


