import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotnine
# import scipy.stats as stats
from plotnine import ggplot, aes,  geom_boxplot, geom_point
from imblearn.pipeline import Pipeline

cust = pd.read_csv("C:\\Users\\SriramvelM\\Downloads\\train.csv")
cust.head(2)
cust.shape

cust['y'] = cust['y'].map({'yes':1,'no':0})

############### Data Cleaning ############################
#Datatypes of dataset
cust.dtypes

#Converting day column from int to string
cust['day'] = cust['day'].astype(str)
# cust.education_qual = cust.education_qual.astype(str)

#Checking for null values in data
cust.isnull().sum()

#Describing the data
cust.describe()

#Check for unknown values percentage
Job = cust.groupby('job').size()
Job_Unknown = (288/45211) * 100

Education = cust.groupby('education_qual').size()
Eduaction_Unknown = (1857/45211) * 100

call_type = cust.groupby('call_type').size()
call_type_Unknown = (13020/45211) * 100

Prev_outcome = cust.groupby('prev_outcome').size()
Prev_outcome_Unknown = (36959/45211) * 100

#In job and education unknown category can be imputed with mode due to low % of unknown values compared to total count.
#Imputation for job and eduaction feature
cust['job'].mode()
cust['job'] = cust['job'].replace(['unknown'], 'blue-collar')

cust['education_qual'].mode()
cust['education_qual'] = cust['education_qual'].replace(['unknown'], 'secondary')
  
#Check for duplicates
cust = cust.drop_duplicates()
cust.shape

#Boxplot to check the spread of the data
col1 = cust.select_dtypes(include='number')
col = cust.select_dtypes(include='object')
for j in col1:
    for i in col:
        ggplot(cust) + aes(x=i, y=j) + geom_boxplot()
    
#Scatterplot before outlier
for i in col1:
    ggplot(cust) + aes(x=i, y='age') + geom_point()

#Identifying and removing outliers
#Checking for outliers
# for i in col1:
#     iqr = cust[i].quantile(0.75) - cust[i].quantile(0.25)
#     Upper_T = cust[i].quantile(0.75) + (1.5 * iqr)
#     Lower_T = cust[i].quantile(0.25) - (1.5 * iqr)
#     Upper_T, Lower_T

iqr1 = cust.dur.quantile(0.75) - cust.dur.quantile(0.25)
Upper_T1 = cust.dur.quantile(0.75) + (1.5 * iqr1)
Lower_T1= cust.dur.quantile(0.25) - (1.5 * iqr1)
Upper_T1,Lower_T1

iqr2 = cust.num_calls.quantile(0.75) - cust.num_calls.quantile(0.25)
Upper_T2 = cust.num_calls.quantile(0.75) + (1.5 * iqr2)
Lower_T2 = cust.num_calls.quantile(0.25) - (1.5 * iqr2)
Upper_T2,Lower_T2

iqr = cust.age.quantile(0.75) - cust.age.quantile(0.25)
Upper_T = cust.age.quantile(0.75) + (1.5 * iqr)
Lower_T = cust.age.quantile(0.25) - (1.5 * iqr)
Upper_T,Lower_T

#Removing outliers
cust.loc[cust.age > Upper_T,'age'] = Upper_T
cust.loc[cust.age < Lower_T,'age'] = Lower_T

cust.loc[cust.dur > Upper_T1,'dur'] = Upper_T1
cust.loc[cust.dur < Lower_T1,'dur'] = Lower_T1

cust.loc[cust.num_calls > Upper_T2,'num_calls'] = Upper_T2
cust.loc[cust.num_calls < Lower_T2,'num_calls'] = Lower_T2

cust.dtypes
cust.describe()

cust['age'] = cust['age'].astype(int)

#Scatterplot after outlier
for i in col1:
    ggplot(cust) + aes(x=i, y='age') + geom_point()

#Feature vs Target variable
for i in col:
    cust.groupby(i)["y"].mean().sort_values(ascending=False).plot(kind='bar')
    plt.show()

#one hot encoding for required categroical variables
dummy_job = pd.get_dummies(cust['job'], prefix='job')
cust = cust.join(dummy_job)

dummy_call_type = pd.get_dummies(cust['call_type'], prefix='call_type')
cust = cust.join(dummy_call_type)

dummy_mon = pd.get_dummies(cust['mon'], prefix='mon')
cust = cust.join(dummy_mon)

dummy_prev_outcome = pd.get_dummies(cust['prev_outcome'], prefix='prev_outcome')
cust = cust.join(dummy_prev_outcome)

#Label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
cust['education_qual'] = label_encoder.fit_transform(cust['education_qual'])
cust['marital'] = label_encoder.fit_transform(cust['marital'])

cust.dtypes
cust['marital'] = cust['marital'].astype(str)
cust['education_qual'] = cust['education_qual'].astype(str)
cust['y'] = cust['y'].astype(str)

#Remove unnecessary columns
cust = cust.drop(['job', 'call_type', 'mon', 'prev_outcome'], axis=1)

cust['education_qual'] = cust['education_qual'].astype(str)
cust['y'] = cust['y'].astype(str)
cust['job_admin.'] = cust['job_admin.'].astype(str)
cust['job_blue-collar'] = cust['job_blue-collar'].astype(str)
cust['job_entrepreneur'] = cust['job_entrepreneur'].astype(str)
cust['job_housemaid'] = cust['job_housemaid'].astype(str)
cust['job_management'] = cust['job_management'].astype(str)
cust['job_retired'] = cust['job_retired'].astype(str)
cust['job_self-employed'] = cust['job_self-employed'].astype(str)
cust['job_services'] = cust['job_services'].astype(str)
cust['job_student'] = cust['job_student'].astype(str)
cust['job_technician'] = cust['job_technician'].astype(str)
cust['job_unemployed'] = cust['job_unemployed'].astype(str)
cust['call_type_cellular'] = cust['call_type_cellular'].astype(str)
cust['call_type_telephone'] = cust['call_type_telephone'].astype(str)
cust['call_type_unknown'] = cust['call_type_unknown'].astype(str)
cust['mon_apr'] = cust['mon_apr'].astype(str)
cust['mon_aug'] = cust['mon_aug'].astype(str)
cust['mon_dec'] = cust['mon_dec'].astype(str)
cust['mon_feb'] = cust['mon_feb'].astype(str)
cust['mon_jan'] = cust['mon_jan'].astype(str)
cust['mon_jul'] = cust['mon_jul'].astype(str)
cust['mon_jun'] = cust['mon_jun'].astype(str)
cust['mon_mar'] = cust['mon_mar'].astype(str)
cust['mon_may'] = cust['mon_may'].astype(str)
cust['mon_nov'] = cust['mon_nov'].astype(str)
cust['mon_oct'] = cust['mon_oct'].astype(str)
cust['mon_sep'] = cust['mon_sep'].astype(str)
cust['prev_outcome_failure'] = cust['prev_outcome_failure'].astype(str)
cust['prev_outcome_other'] = cust['prev_outcome_other'].astype(str)
cust['prev_outcome_success'] = cust['prev_outcome_success'].astype(str)
cust['prev_outcome_unknown'] = cust['prev_outcome_unknown'].astype(str)

#Seperating dependent and independent variables
x = cust.drop('y', axis='columns')
y = cust['y']

from sklearn .model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.80, random_state=3)

np.shape(x_train)
np.shape(x_test)
np.shape(y_train)
np.shape(y_test)

x_train.dtypes
# cont = [0,4,5]
# cat = [1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]

cont = x_train.select_dtypes(include='number').columns
cat = x_train.select_dtypes(include='object').columns

#Scaling of data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

t = [('n', StandardScaler(), cont)]
selective = ColumnTransformer(transformers=t)
model = LogisticRegression()

#From countplot we can see that our target in imbalanced. So, applyiing imbalanced learning before modelling
#Imbalanced learning
from imblearn.combine import SMOTEENN
from collections import Counter
smote = SMOTEENN(sampling_strategy='all')
x_sm, y_sm = smote.fit_resample(x_train,y_train)
counter = Counter(y_sm)
print(counter)

pipeline = Pipeline([('s',selective),('sampling', SMOTEENN()),('m',model)])

# define the evaluation procedure
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#Hyperparameters for logistic regression
solver = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

param_grid = { 'm__penalty': penalty, 'm__C': c_values}

#GridsearchCV to get best parameters
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid=param_grid,cv=cv, scoring='accuracy',error_score=0)
grid_search.fit(x_train, y_train)

#Best fit
grid_search.best_params_

# evaluate the model
from sklearn.model_selection import cross_val_score
m_scores = cross_val_score(pipeline, x_train, y_train, scoring='accuracy', cv=cv)

# summarize the result
from numpy import mean
from numpy import std
print('Accuracy: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))

#Fit the best model
model = LogisticRegression(penalty='l2', C=0.01)
pipeline = Pipeline([('s',selective),('sampling', SMOTEENN()),('m',model)])
pipeline.fit(x_train, y_train)
y_pred=pipeline.predict(x_test)

from sklearn.metrics import roc_auc_score
pipeline.score(x_test, y_test)
roc_auc_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()

TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
sensitivity
specificity=TN/float(TN+FP)
specificity

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',
'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',
'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',
'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',
'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',
'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',
'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',
'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)

########### this data performs poorly on logistic regression model.###########

############# KNN algorithm ###################################
from sklearn.neighbors import KNeighborsClassifier
# define models and parameters
model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

#pipeline for KNN
pipeline = Pipeline([('s',selective),('sampling', SMOTEENN()),('m',model)])

# define grid search
param_grid = dict(m__n_neighbors=n_neighbors,m__weights=weights,m__metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#Fitting the best model
model = KNeighborsClassifier(n_neighbors=1, weights='distance', metric='manhattan')
pipeline = Pipeline([('s',selective),('sampling', SMOTEENN()),('m',model)])
pipeline.fit(x_train, y_train)
y_pred=pipeline.predict(x_test)

from sklearn.metrics import roc_auc_score
pipeline.score(x_test, y_test)
roc_auc_score(y_test, y_pred)

#Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()

TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
sensitivity
specificity=TN/float(TN+FP)
specificity

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',
'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',
'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',
'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',
'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',
'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',
'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',
'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)

################# Support Vector Machine ##################################
from sklearn.svm import SVC

# define model and parameters
model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']

#define pipeline
pipeline = pipeline = Pipeline([('s',selective),('sampling', SMOTEENN()),('m',model)])

# define grid search
grid = dict(m__kernel=kernel,m__C=C,m__gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=pipeline, param_grid=grid, cv=cv, verbose=2, n_jobs=4, scoring='accuracy')
grid_result = grid_search.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

########## Fitting the best model ##############################
model = SVC(C=0.01, gamma='scale', kernel='poly')
pipeline = pipeline = Pipeline([('s',selective),('sampling', SMOTEENN()),('m',model)])
pipeline.fit(x_train, y_train)
y_pred=pipeline.predict(x_test)

from sklearn.metrics import roc_auc_score
pipeline.score(x_test, y_test)
roc_auc_score(y_test, y_pred)

#Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()

TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
sensitivity
specificity=TN/float(TN+FP)
specificity

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',
'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',
'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',
'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',
'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',
'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',
'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',
'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)
