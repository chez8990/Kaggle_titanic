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

print cleaned_prefix

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
# 	cleaned_prefix.append('Non alpha')

# sns.barplot(x,y)
# plt.show()



def one_hot(classList, return_labels=True):
	precode = LabelEncoder()
	transformed = precode.fit_transform(classList)
	if return_labels ==True:
		return transformed, precode.classes_
	return transformed

gender = pd.get_dummies(train['Sex']) #performs one hot
embarkation = pd.get_dummies(train['Embarked'])

train = train.drop(['Sex', 'Embarked'], axis=1)
train = train.join([gender, embarkation])
train = train.rename(columns={'female':'Female', 'male':'Male', 'C':'Churberg', 'Q': 'Queenstown', 'S':'Southampton'})

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
		survived = float(train[train['Ticket'].str.contains('|'.join(matches))]['Survived'].sum())/float(train['Survived'].count())
		print survived
		survived_count.append(survived)
	return survived_count

sns.barplot(cleaned_prefix, PrefixSurvival(cleaned_prefix))
plt.show()

print trainp