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
for raw in train['Ticket']:
	ticket_split= raw.split(' ')
	if len(ticket_split) >1:
		prefix = ticket_split[0].replace("/", '').replace('.','')
		if raw not in prefix_dict:
			prefix_dict[raw]=prefix
		if prefix not in cleaned_prefix:
			cleaned_prefix.append(prefix)

# Calculate the means of the tickets with the same prefix
def ticket_means(prefix_list):
	clean_means=[]
	for pre in prefix_list:
		single_sum, single_length= 0,0 
		matches = [x for x in prefix_dict if prefix_dict[x]==pre]
		num_match = len(matches)
		for instance in matches:
			find_match = train[train['Ticket'].str.contains(instance)]
			single_sum += find_match['Fare'].sum() 
			single_length+=find_match.shape[0]
		clean_means.append(single_sum/single_length)
	# clean_means.append(train[train['Ticket'].str.isdigit()]['Fare'].mean())
	return clean_means

x = cleaned_prefix 
y = ticket_means(cleaned_prefix)


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

fig, axes = plt.subplots(5,6, figsize=(20,15))
fig.legend_out = True

# def graphPortTickets(prefix_list):

