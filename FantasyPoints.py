# -*- coding: utf-8 -*-

import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import math

from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F

year1 = 2016
year2 = year1+1

#upload files
dfyear1 = pd.read_csv('player'+str(year1)+'.csv')
dfyear2 = pd.read_csv('player'+str(year2)+'.csv')
def clean_year(year, scoring):
    #upload files
    df = pd.read_csv('player'+str(year)+'.csv')
    
    #add column for fantasy points, make dictionary to relay positions, list all IDs
    df['Fantasy Points'] = add_fantasy_points(df)


    posdict = {df['Player ID'].iloc[i]: df['Position'].iloc[i]
                for i in range(len(df))}
    namedict = {df['Player ID'].iloc[i]: df['Name'].iloc[i]
                for i in range(len(df))}
    
    year_game_data = game_data(df)
    
    
    
    year_game_data = year_game_data[['Rushing Att', 'Rushing Yds', 'Rushing Y/A',
                        'Rushing TD', 'Receiving Tgt', 'Receiving Rec',
                        'Receiving Yds', 'Receiving Y/R', 'Receiving TD',
                        'Age', 'Games Played', 'Receiving Y/Tgt', 'Scoring TD',
                        'Scoring Pts',	'Fumbles Fmb', 'Fumbles FL',
                        'Fumbles FF', 'Fumbles FR', 'Fumbles Yds', 'Fumbles TD',
                        'Off. Snaps Num', 'Kick Returns Rt', 'Kick Returns Yds',
                        'Kick Returns Y/Rt', 'Kick Returns TD', 'Scoring 2PM',
                        'Punt Returns Ret', 'Punt Returns Yds',
                        'Punt Returns Y/R', 'Punt Returns TD', 'Passing Cmp',
                        'Passing Att', 'Passing Yds', 'Passing TD', 'Passing Int',
                        'Passing Rate', 'Passing Sk', 'Passing Yds.1',
                        'Passing Y/A', 'Passing AY/A', 'Fantasy Points'
                        ]]
    
    return year_game_data
    
# define fantasy scoring
def add_fantasy_points(df, scoring = 'PPR'):
    
    points = (df['Rushing Yds'] + df['Receiving Yds'])*0.1 + \
    (df['Rushing TD'] + df['Receiving TD'] + df['Fumbles TD'] + 
         df['Kick Returns TD'] + df['Def Interceptions TD'])*6 + \
    df['Passing Yds']*0.04 + \
    df['Passing TD']*4 + \
    (df['Passing Int'] + df['Fumbles FL'])*-2 + \
    df['Scoring 2PM']*2
    
    if scoring == 'PPR':
        points = points + df['Receiving Rec']
    elif scoring == 'hPPR':
        points = points + df['Receiving Rec'] *0.5
    else:
        print('No points per reception')
        
    return points

#find the intersection of two lists
def intersection(list1, list2):
    list1_as_set = set(list1)
    intersection = list1_as_set.intersection(list2)
    
    intersection_as_list = list(intersection)
    
    return intersection_as_list

#add column for fantasy points, make dictionary to relay positions, list all IDs
dfyear2['Fantasy Points'] = add_fantasy_points(dfyear2)


posdictyear2 = {dfyear2['Player ID'].iloc[i]: dfyear2['Position'].iloc[i]
            for i in range(len(dfyear2))}
namedictyear2 = {dfyear2['Player ID'].iloc[i]: dfyear2['Name'].iloc[i]
            for i in range(len(dfyear2))}


dfyear1['Fantasy Points'] = add_fantasy_points(dfyear1)
posdictyear1 = {dfyear1['Player ID'].iloc[i]: dfyear1['Position'].iloc[i]
            for i in range(len(dfyear1))}
namedictyear1 = {dfyear1['Player ID'].iloc[i]: dfyear1['Name'].iloc[i]
            for i in range(len(dfyear1))}

dictname = namedictyear2
dictname.update(namedictyear1)

dictpos = posdictyear2
dictpos.update(posdictyear1)

fantasy_seasonyear2 = dfyear2.groupby('Player ID')['Fantasy Points'].sum()
fantasy_seasonyear1 = dfyear1.groupby('Player ID')['Fantasy Points'].sum()

between = pd.DataFrame([fantasy_seasonyear1,fantasy_seasonyear2], index = ['{}'.format(year1), '{}'.format(year2)]).T

positions = ['WR', 'RB', 'TE', 'QB', 'None']
poslist = []
namelist = []

for pos in between.index:
    if dictpos[pos] != dictpos[pos]:
        dictpos[pos] = 'None'
    else:
        pass
    poslist.append(dictpos[pos])
    namelist.append(dictname[pos])
   
between['Position'] = poslist                   
between['Name'] = namelist

'''
fig1 = px.scatter(between, '{}'.format(year1), '{}'.format(year2), hover_name = 'Name',
                 color = 'Position', color_discrete_sequence=('red', 'green', 'black', 'blue', 'silver'))

plot(fig1)
'''

#top40TE = between[between['Position'] == 'WR'].nlargest(40, 'year2')

#fig = px.scatter(top40TE, '{}'.format(year1), ''.format(year2), hover_name = 'Name',
#                 color = 'Position', color_discrete_sequence=('red', 'green', 'black', 'orange', 'silver'))

#print(between)

between.to_csv('Total_fantasy.csv')

yeardif = {}
for name in list(between.index):
    try:
        score1 = fantasy_seasonyear1[name]
        score2 = fantasy_seasonyear2[name]
        if score1 != 'NaN':
            if score2 != 'NaN':
                ave = (score1+score2)/2
            else:
                ave = score1/2
        else:
            ave = score2/2
        yeardif[name] = [(score2-score1), score1, score2, ave]
    except:
        pass


scoredf = pd.DataFrame(yeardif, index  = ['Dif', 'Year1', 'Year2', 'Average']).T
posdf = pd.DataFrame.from_dict(dictpos, orient='index', columns = ['Position'])
namedf = pd.DataFrame.from_dict(dictname, orient='index', columns = ['Name'])
difdf = pd.concat([scoredf,posdf,namedf], axis = 1)

#select dataframe only where a player scored points in both seasons
difdf = difdf[~difdf.isnull().any(axis = 1)]
#print(fantasy_seasonyear2[between['Name'].index[0]])
'''
fig2 = px.scatter(difdf, 'Dif', 'Average', hover_name = 'Name',
                 color = 'Position',
                 color_discrete_sequence=('red', 'green',
                                          'black', 'blue', 'silver'),
                 title = 'Change from '+str(year1)+' to '+str(year2))

plot(fig2)
'''
def game_data(df):
    
    df = df.drop(['Unnamed: 0','Date','G#','Week','Away','Result','Season'], axis = 1)
    age = df.groupby('Player ID')['Age'].min().rename('Age')
    games_played = df.groupby('Player ID')['Rk'].max().rename('Games Played')
    # Include this by assigning each team a position in a one-hot incoder and summing
    df = df.drop(['Tm','Opp'], axis = 1)
    
    df = df.drop(['Age','Rk'], axis = 1)
    game_data = df.groupby('Player ID').sum(numeric_only = True)
    # Include with one hot encoder
    #pos = df.groupby('Player ID')['Position'].rename('Position')
    
    game_data = game_data.join([age,games_played])
    
    return game_data

#year1_game_data = game_data(dfyear1)
#year2_game_data = game_data(dfyear2)
#print(year1_game_data.head(5))



year1_game_data = clean_year(year1, 'PPR')
#print(year1_game_data.head())
year2_game_data = clean_year(year2, 'PPR')
year3_game_data = clean_year(year2+1, 'PPR')


common_players1 = intersection(year1_game_data.index.to_list(),
                              year2_game_data.index.to_list())

common_players2 = intersection(year2_game_data.index.to_list(),
                              year3_game_data.index.to_list())

year1_game_data.loc[common_players1].to_csv('test2.csv')
year2_game_data.loc[common_players2].to_csv('test3.csv')

#print(year2_game_data.loc[common_players2])

X = year1_game_data.loc[common_players1].to_numpy()
X = np.append(X,
              year2_game_data.loc[common_players2].to_numpy(),
              axis = 0)
y = year2_game_data.loc[common_players1]['Fantasy Points'].to_numpy()
y = np.append(y,
              year3_game_data.loc[common_players2]['Fantasy Points'].to_numpy(),
              axis = 0)


X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
print(X.shape)

## NN model
'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.hidden1 = nn.Linear(56, 20)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(20, 5)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(5, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))

        return x
'''
net = nn.Sequential(
    nn.Linear(41, 30),
    nn.ReLU(),
    nn.Linear(30, 20),
    nn.ReLU(),
    nn.Linear(20, 1),
    nn.ReLU()
    )

#net = Net()
#print(net)

MSEloss = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

n_epochs = 100
batch_size = 40
 
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = net(Xbatch)
        ybatch = y[i:i+batch_size]
        if i ==len(X)-50:
            print(Xbatch, y_pred, ybatch)
        else:
            pass
        loss = MSEloss(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

year4_game_data = clean_year(year2+2, 'PPR')

common_players3 = intersection(year3_game_data.index.to_list(),
                              year4_game_data.index.to_list())

X_test = year3_game_data.loc[common_players3].to_numpy()
y_test = year4_game_data.loc[common_players3]['Fantasy Points'].to_numpy()

X_test = torch.tensor(X_test, dtype=torch.float32)

y_pred_test = net(X_test)

fig3 = px.scatter(x = y_test.reshape(-1, 1).T[0], y = list(y_pred_test.detach().numpy().T[0]),
                 title = str(year2+1)+' to '+str(year2+2))

plot(fig3)
