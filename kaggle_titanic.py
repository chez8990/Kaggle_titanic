from datetime import datetime as dt
from collections import OrderedDict
import re
import math
import datetime
import seaborn as sns

import pylab
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pylab as pylab

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression 

from sklearn import datasets, tree
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, RFE
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print pd.__version__ 

col_meanings = [
    'The ID of the passenger',
    'Did the passenger survive ? 1 = Yes, 0 = No',
    'Ordinal Value for passenger class, 1 being the highest',
    'Name',
    'Gender',
    'Age',
    'Passenger\'s siblings and spouses on board with',
    'Passenger\'s parents and children on board',
    'Ticket Number',
    'Passenger Fare',
    'Cabin Number',
    'Port of Embarkation'
]

data_dict = pd.DataFrame({
	'Attribute': train.columns,
	'Type': [train[col].dtype for col in train.columns],
	'Meaning': col_meanings,
	'Example': [train[col].iloc[2] for col in train.columns]

	})

prefix_dict=OrderedDict([])
cleaned_prefix=[]


for raw in [item.split(' ')[0] for item in [pre for pre in train['Ticket'].value_counts().index.tolist() if not pre.isalnum()]]:
    cleaned = re.sub(r'\W+', '', raw)
    
    if raw not in prefix_dict:
        prefix_dict[raw] = cleaned
        
    if cleaned not in cleaned_prefix:
        prefix_dict[cleaned] = raw
        cleaned_prefix.append(cleaned)


def ticket_means(prefix_list):
	clean_means=[]
	for pre in prefix_list:
		matches = [x for x in prefix_dict if prefix_dict[x]==pre]
		before = train[train['Ticket'].str.contains("|".join(matches))]['Fare'].mean()
		clean_means.append(before)
	clean_means.append(train[train['Ticket'].str.isdigit()]['Fare'].mean())
	return clean_means


x = cleaned_prefix 
y = ticket_means(cleaned_prefix)
# if 'Non alpha' not in cleaned_prefix:
# 	x.append('Non alpha')

# sns.barplot(x,y)
# plt.show()



def one_hot(classList, return_labels=True):
	precode = LabelEncoder()
	transformed = precode.fit_transform(classList)
	if return_labels ==True:
		return transformed, precode.classes_
	return transformed

train_gender = pd.get_dummies(train['Sex']) #performs one hot
test_gender = pd.get_dummies(test['Sex'])
train_embarkation = pd.get_dummies(train['Embarked'])
test_embarkation = pd.get_dummies(test['Embarked'])

train = train.drop(['Sex', 'Embarked'], axis=1)
train = train.join([train_gender, train_embarkation])
train = train.rename(columns={'female':'Female', 'male':'Male', 'C':'Churberg', 'Q': 'Queenstown', 'S':'Southampton'})

test= test.drop(['Sex', 'Embarked'], axis=1)
test = test.join([test_gender, test_embarkation])
test = test.rename(columns={'female':'Female', 'male':'Male', 'C':'Churberg', 'Q': 'Queenstown', 'S':'Southampton'})


# fig, axes = plt.subplots(5,6, figsize=(20,15))
# fig.legend_out = True


#for each prefix, graph the percentage of 
def graphPortTickets(prefix_list):
	col, row, loop = (0,0,0)
	lookup=[]
	for pre in prefix_list:
		row = int(math.floor(loop/6))

		matches = [x for x in prefix_dict if prefix_dict[x]==pre]
		df = train[train['Ticket'].str.contains("|".join(matches))]
		df_c = float(df['Churberg'].sum())/float(df['PassengerId'].count())
		df_q = float(df['Queenstown'].sum())/float(df['PassengerId'].count())		
		df_s = float(df['Southampton'].sum())/float(df['PassengerId'].count())		

		x = ['Churberg', 'Queenstown', 'Southampton']
		y = [df_c,df_q, df_s]
		if row==4 and col==5:
			ax = sns.barplot(x,y,hue=x,ax=axes[row,col])
		else:
			ax = sns.barplot(x,y,ax=axes[row,col])
		col+=1
		loop+=1
		ax.set_xticks([])

		if col==6:
			col =0
		axes[row,col].set_title('-{}- by port'.format(pre))



# graphPortTickets(cleaned_prefix)
# plt.legend(bbox_to_anchor=(1.05, 5.8), loc=2, borderaxespad=0.)
# plt.show()


def PrefixSurvival(prefix_list):
	survived_count=[]
	for pre in prefix_list:
		matches = [x for x in prefix_dict if prefix_dict[x]==pre]
		candidate = train['Ticket'].str
		survived = float(train[candidate.contains('|'.join(matches))]['Survived'].sum())/float(train[candidate.contains('|'.join(matches))]['Survived'].count())
		survived_count.append(survived)
	survived_count.append(float(train[candidate.isdigit()]['Survived'].sum())/float(train[candidate.isdigit()]['Survived'].count()))
	return survived_count


new_x = cleaned_prefix + ['Non_alpha']

# sns.barplot(new_x, PrefixSurvival(cleaned_prefix))
# plt.xticks(rotation=30)

def golden_ticket(row):
	golden_list = ['PC', 'PP', 'F.C', 'FC']
	if filter(lambda x: x in row['Ticket'], golden_list):
		return 1
	else:
		return 0
	return row['Ticket'].str.contains('PC')

train['Golden ticket'] = train.apply(lambda x: golden_ticket(x), axis =1 )
test['Golden ticket'] = test.apply(lambda x: golden_ticket(x), axis =1)

train = train.drop('Ticket', axis=1)
test = test.drop('Ticket', axis=1)

cabin_dict = {
	'A': 1,
	'B': 2,
	'C': 3,
	'D': 4,
	'E': 5,
	'F': 6,
	'G': 7,
	'T': 8
}

train['Cabin'] = train['Cabin'].fillna(value='G')
test['Cabin'] = test['Cabin'].fillna(value='G')

train['Cabin_ord'] = train.apply(lambda x: cabin_dict[x['Cabin'][0]], axis =1)
test['Cabin_ord'] = test.apply(lambda x: cabin_dict[x['Cabin'][0]], axis =1)

train = train.drop(['Cabin', 'PassengerId'], axis =1)
test = test.drop(['Cabin'], axis=1)

def refineName(row):
	suffix_list = ['Mrs', 'Mr.', 'Miss', 'Master']
	suffix = filter(lambda x: x in row['Name'], suffix_list)
	if suffix:
		return suffix[0]
	else:
		return 'Other'

train['Name'] = train.apply(lambda x: refineName(x), axis=1)
test['Name'] = test.apply(lambda x: refineName(x), axis=1)

suffixes = pd.get_dummies(train['Name'])
test_suffixed = pd.get_dummies(test['Name'])

train = train.join(suffixes)
train = train.drop('Name', axis=1)

test = test.join(suffixes)
test = test.drop('Name', axis=1)

test['Fare'] = test['Fare'].fillna(value=0)


#Prepareing data for regression, fill up age void
with_age = train[train['Age']>0]
no_age = train[train['Age'].isnull()]
no_age = no_age.drop('Age', axis=1)

test_with_age = test[test['Age']>0]
test_no_age = test[test['Age'].isnull()].drop('Age', axis=1)

#instantiate model
train_model = LinearRegression()
test_model = LinearRegression()

train_fitted = train_model.fit(with_age.drop('Age', axis=1), with_age['Age'])
test_fitted = test_model.fit(test_with_age.drop('Age', axis=1), test_with_age['Age'])

train_prediction = train_model.predict(no_age)
test_prediciton = test_model.predict(test_no_age)

#replace null values with prediction
no_age['Age'] = train_prediction
test_no_age['Age'] = test_prediciton 

train = with_age.append(no_age)
test = test_with_age.append(test_no_age)

# correlation heat map



def rich(row):
	if row['Fare']>= 50:
		return 1 
	else:
		return 0

def isKid(row):
	if row['Age']<=6:
		return 1
	else:
		return 0
def hasFam(row):
	if row['SibSp']>0 or row['Parch']>0:
		return 1

	else:
		return 0

train['Rich'] = train.apply(lambda x: rich(x), axis=1)
train['isKid'] = train.apply(lambda x: isKid(x), axis=1)
train['hasFam'] = train.apply(lambda x: hasFam(x), axis=1)

test['Rich'] = test.apply(lambda x: rich(x), axis=1)
test['isKid'] = test.apply(lambda x: isKid(x), axis=1)
test['hasFam'] = test.apply(lambda x: hasFam(x), axis=1)


train = train.drop(['Male', 'Fare', 'Parch', 'SibSp'], axis = 1)
test = test.drop(['Male', 'Fare', 'Parch', 'SibSp'], axis =1)


# plt.figure(figsize=(15,15))
# plt.title('Feature correlations')
# sns.heatmap(train.corr(), linewidths = 0.1, vmax=1.0, cmap=plt.cm.seismic, annot=True, square=True)
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# plt.show()


from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(n_estimators = 1000)
gaus = GaussianNB()
logreg = LogisticRegression()
dtree = DecisionTreeClassifier()
svc_rbf = SVC(kernel = 'rbf')
svc_lin = SVC(kernel = 'linear')
knn = KNeighborsClassifier(n_neighbors = 3)
per = Perceptron()
grd = GradientBoostingClassifier()

y = train['Survived'].values
x = train.drop('Survived', axis=1)

X_train, X_test, y_train, y_test =  train_test_split(x,y,test_size=0.3, stratify= y)
# for _ in range(3000):
# 	grd.fit(X_train,y_train)
# 	grd_score = grd.score(X_test,y_test)

# 	if grd.score>0.88:
# 		algorithms = [{'algo': rf, 'color': '#4285f4', 'name': 'Random Forest'}, {'algo': gaus, 'color': 'red', 'name': 'Gaussian'}, 
# 		{'algo': logreg, 'color': 'blue', 'name': 'Logistic Regressions'},{'algo': dtree, 'color': 'orange', 'name': 'Decision Tree'}, 
# 		{'algo': svc_rbf, 'color': 'lime', 'name': 'SVC-RBF'}, {'algo': svc_lin, 'color': 'purple', 'name': 'Linear SVC'},
# 		{'algo': knn, 'color': 'yellow', 'name': 'KNN'},{'algo': per, 'color': 'indigo', 'name': 'Perceptron'}, 
# 		{'algo': grd, 'color': 'black', 'name': 'Gradient Boosting'}
# 		]

# 		for alg in algorithms:
# 			algo = alg['algo']
# 			algo.fit(X_train, y_train)
# 			predictions = algo.predict(X_test)
# 			fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
# 			auc = metrics.auc(fpr, tpr)
# 			plt.plot(fpr, tpr, alg['color'], label = '{} AUC = {:.2f}'.format(alg['name'],auc))

# 		plt.title('ROC')
# 		plt.legend(loc = 'lower right')
# 		plt.plot([0,1],[0,1],'r--')
# 		plt.xlabel("True Positive Rate")
# 		plt.ylabel("False Positive Rate")
# 		plt.show()

# 		scores = [algorithm.score(X_test, y_test) for algorithm in [al['algo'] for al in algorithms ]]
# 		scoring_df = pd.DataFrame({'Algorithms':[al['name'] for al in algorithms], 'Score':scores})

# 		print scoring_df
# 		break


rfe = RFE(estimator=grd, n_features_to_select=1, step=1)
rfe.fit(x,y)
ranking = rfe.ranking_


# print pd.DataFrame(list(zip(x.columns, ranking)), columns=['Feat', 'Ranking'])

#optimizing grd 
# lrate = [float(i)/float(1000) for i in range(1,1000)]

# grd_scores =[]

# for mu in lrate:
# 	print mu
# 	grd = GradientBoostingClassifier(n_estimators =17, learning_rate= mu)
# 	grd.fit(X_train,y_train)
# 	grd_preds = grd.predict(X_test)
# 	grd_fpr, grd_tpr, threshold = metrics.roc_curve (y_test, grd_preds)
# 	grd_auc = metrics.auc(grd_fpr, grd_tpr)
# 	grd_score = grd.score(X_test, y_test)
# 	grd_scores.append({'Criteria': mu, 'Score': grd_score})

# grd_scores = sorted(grd_scores, key= lambda k: k['Score'], reverse=True)

# print grd_scores[0]

optimal_learning_rate = 0.054


#Feature engineering for test set before predictions




grd = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.054)
grd.fit(X_train,y_train)


new_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

new_df['PassengerId'] = test['PassengerId'].values

test = test.drop('PassengerId', axis=1)

new_df['Survived'] = grd.predict(test)

new_df.to_csv('titanic_final.csv', index=False)