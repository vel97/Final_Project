

# cust['y'] = cust['y'].map({'yes':1,'no':0})

################# Data Cleaing ################################
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

#Since 28% of unknown values are in call_type feature we need to check feature vs Target EDA for
#decision making

#81 % of category is unkown for pre_outcome feature, it might be significant for our classification. So,
#we continue with it.

#Check for duplicates
cust = cust.drop_duplicates()
cust.shape

############################## EDA #########################################
#Countplot to check the freq distributiion of our target and other variable
import seaborn as sns
sns.countplot(data = cust, x='y')
plt.show()

# facet_grid(facets="~y")
col = cust.select_dtypes(include='object')
for i in col:
    ggplot(cust) + aes(x=i, y="dur")+ labs(x=i,y="dur") + geom_bar(stat = 'identity')
    plt.show()

for i in col:
    ggplot(cust) + aes(x=i, y="age")+ labs(x=i,y="age") + geom_bar(stat = 'identity')
    plt.show()

for i in col:
    ggplot(cust) + aes(x=i, y="num_calls")+ labs(x=i,y="num_calls",) + geom_bar(stat = 'identity')
    plt.show()


#Histogram plot
col1 = cust.select_dtypes(include='number')
for i in col1:
    ggplot(cust) + aes(x=i) + geom_histogram()
    plt.show()

#kdeplot of histogram to check the distribution of customer reaction('y') with respect to age
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

cust['y'] = label_encoder.fit_transform(cust['y'])

for i in col1:
    fig, ax = plt.subplots(1,1)
    sns.kdeplot(cust[cust["y"]==1][i], fill=True, color="blue", label="+ve", ax=ax)
    sns.kdeplot(cust[cust["y"]==0][i], fill=True, color="green", label="-ve", ax=ax)
    ax.set_xlabel(i)
    ax.set_ylabel("Y")
    fig.suptitle(i +" "+"vs Positive")
    ax.legend()
    fig.show()

#Boxplot to check the spread of the data
for j in col1:
    for i in col:
        ggplot(cust) + aes(x=i, y=j) + geom_boxplot()
    

#Scatter plot to check replationship between continuous independent variables
for i in col1:
    ggplot(cust) + aes(x=i, y='age') + geom_point()


#Feature vs Target variable
for i in col:
    cust.groupby(i)["y"].mean().sort_values(ascending=False).plot(kind='bar')
    plt.show()


# label_encoder object knows how to understand word labels.
# For tree algorithms label encoding is sufficient for model to understand our data.
#No need of one-hot encoding.
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

cust['y'] = label_encoder.fit_transform(cust['y'])
cust['education_qual'] = label_encoder.fit_transform(cust['education_qual'])
cust['marital'] = label_encoder.fit_transform(cust['marital'])
cust['mon'] = label_encoder.fit_transform(cust['mon'])
cust['prev_outcome'] = label_encoder.fit_transform(cust['prev_outcome'])
cust['job'] = label_encoder.fit_transform(cust['job'])
cust['call_type'] = label_encoder.fit_transform(cust['call_type'])

#Adjusting the datatype to str as they changed to int during label encoding.
cust['job'] = cust['job'].astype(str)
cust['marital'] = cust['marital'].astype(str)
cust['mon'] = cust['mon'].astype(str)
cust['education_qual'] = cust['education_qual'].astype(str)
cust['prev_outcome'] = cust['prev_outcome'].astype(str)
cust['y'] = cust['y'].astype(str)
cust['call_type'] = cust['call_type'].astype(str)

#Splitting the dataset as train and test
x = cust.drop('y', axis='columns')
y = cust['y']

#Train test split
from sklearn .model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state=0)

np.shape(x_train)
np.shape(x_test)
np.shape(y_train)
np.shape(y_test)

#From countplot we can see that our target in imbalanced. So, applyiing imbalanced learning before modelling
#Imbalanced learning
from imblearn.combine import SMOTEENN
from collections import Counter
smote = SMOTEENN(sampling_strategy='all')
x_sm, y_sm = smote.fit_resample(x_train,y_train)
counter = Counter(y_sm)
print(counter)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

#Fitting the tree
fit = dt.fit(x_train, y_train)
fit = dt.fit(x_sm, y_sm)

#Evaluation of decision treee using AUROC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

#Predicting test data
y_pred = dt.predict(x_test)

#Computing test data accuracy
acc = accuracy_score (y_test, y_pred)
print(acc)
auc = roc_auc_score(y_test, y_pred)
print(auc)

########## making pipeline for smote enn and decision tree #################################
steps = [('sampling', SMOTEENN(sampling_strategy='all')), ('model', DecisionTreeClassifier())]
pipe = Pipeline(steps=steps)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipe, x_train, y_train, cv=cv)
print('Mean ACCURACY: %.3f' % mean(scores))

##GridsearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2 ,3 ,5 , 6 ,7],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 4, 5],
    'max_leaf_nodes': [None, 5, 10, 20],
}

dtree_cv = GridSearchCV(dt,  param_grid=param_grid, cv=5, verbose=2, n_jobs=4)
dtree_cv.fit(x_train, y_train)

#Best params
dtree_cv.best_params_

################# Best Model & Feature Importances ########################
steps = [('sampling', SMOTEENN(sampling_strategy='all')), ('model', DecisionTreeClassifier(max_depth= 7, criterion='gini', min_samples_leaf=5, min_samples_split=5))]
pipe = Pipeline(steps=steps)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
y_predict_prob = pipe.predict_proba(x_test)
Acc_dt = pipe.score(x_test, y_test)
roc_dt = roc_auc_score(y_test, y_pred)
f1_dt = f1_score(y_test, y_pred, average = 'macro')
Feature_Imp = pipe['model'].feature_importances_ 
plt.bar([i for i in range(len(Feature_Imp))],Feature_Imp)
plt.show()
FI = list(zip(Feature_Imp,x_test.columns))
df = pd.DataFrame(FI)
df

################## tree ##########################################
from sklearn import tree
# dt = DecisionTreeClassifier(max_depth = 7)
tree.plot_tree(pipe['model'])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (4,4), dpi = 300)
tree.plot_tree(pipe['model'], feature_names=x_test.columns, filled=True)
fig.savefig('dtree.png')

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


############# random forest classifier #############################
########## making pipeline for smote enn and random forest #################################
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

#Creating pipeline
steps = [('sampling', SMOTEENN(sampling_strategy='all')), ('model', RandomForestClassifier())]
pipe = Pipeline(steps=steps)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipe, x_train, y_train, cv=cv)
print('Mean ACCURACY: %.3f' % mean(scores))

#Fitting random model
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier()
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
roc_auc_score(y_test, y_pred)

#Hyper parameter tuning of random forest to get the best one
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
max_depth = [6,7,8]
min_samples_split = [2,4,5]
min_samples_leaf = [1,2,3]
bootstrap = [True, False]

#Pram grid creation
param_grid = {'n_estimators':n_estimators,'max_depth': max_depth,
'min_samples_split': min_samples_split,'min_samples_leaf': min_samples_leaf,'bootstrap': bootstrap}

#Fitting though gridsearchcv
rf_grid = GridSearchCV(rfc, param_grid=param_grid, cv=10, verbose=2, n_jobs=4)
try:
    rf_grid.fit(x_train, y_train)
except ValueError:
    pass

#Best parameters
rf_grid.best_params_
rf_grid.score(x_train, y_train)
rf_grid.score(x_test, y_test)

#Fitting the best model from best_params
steps = [('sampling', SMOTEENN(sampling_strategy='all')), ('model', RandomForestClassifier(bootstrap=True, max_depth=8, min_samples_leaf=3, min_samples_split=2
, n_estimators=70))]
pipe = Pipeline(steps=steps)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
acc_rf = pipe.score(x_test, y_test)
roc_rf = roc_auc_score(y_test, y_pred)
f1_rf = f1_score(y_test, y_pred, average='macro')
Feature_Imp = pipe['model'].feature_importances_ 
plt.bar([i for i in range(len(Feature_Imp))],Feature_Imp)
plt.show()
FI = list(zip(Feature_Imp,x_test.columns))
df = pd.DataFrame(FI)
df

#Confusion matrix
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

#summary of models
df=pd.DataFrame()
df['Model']=pd.Series(['Decision Tree','Random Forest'])
df['Accuracy']=pd.Series([Acc_dt, acc_rf])
df['ROC_AUC_Score']=pd.Series([roc_dt, roc_rf])
df['F1-Score'] = pd.Series([f1_dt, f1_rf])
df.set_index('Model')
