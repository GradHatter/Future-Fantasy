# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:30:06 2020

@author: jkora
"""
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import math

from sklearn.linear_model import LinearRegression

year1 = 2017
year2 = year1+1

#upload files
df2018 = pd.read_csv('player'+str(year1)+'.csv')
df2019 = pd.read_csv('player'+str(year2)+'.csv')

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
df2019['Fantasy Points'] = add_fantasy_points(df2019)


posdict2019 = {df2019['Player ID'].iloc[i]: df2019['Position'].iloc[i]
            for i in range(len(df2019))}
namedict2019 = {df2019['Player ID'].iloc[i]: df2019['Name'].iloc[i]
            for i in range(len(df2019))}


df2018['Fantasy Points'] = add_fantasy_points(df2018)
posdict2018 = {df2018['Player ID'].iloc[i]: df2018['Position'].iloc[i]
            for i in range(len(df2018))}
namedict2018 = {df2018['Player ID'].iloc[i]: df2018['Name'].iloc[i]
            for i in range(len(df2018))}

dictname = namedict2019
dictname.update(namedict2018)

dictpos = posdict2019
dictpos.update(posdict2018)

fantasy_season2019 = df2019.groupby('Player ID')['Fantasy Points'].sum()
fantasy_season2018 = df2018.groupby('Player ID')['Fantasy Points'].sum()

between = pd.DataFrame([fantasy_season2019,fantasy_season2018], index = ['2019', '2018']).T

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

fig = px.scatter(between, '2018', '2019', hover_name = 'Name',
                 color = 'Position', color_discrete_sequence=('red', 'green', 'black', 'blue', 'silver'))

#top40TE = between[between['Position'] == 'WR'].nlargest(40, '2019')

#fig = px.scatter(top40TE, '2018', '2019', hover_name = 'Name',
#                 color = 'Position', color_discrete_sequence=('red', 'green', 'black', 'orange', 'silver'))

#print(between)

yeardif = {}
for name in list(between.index):
    try:
        score1 = fantasy_season2018[name]
        score2 = fantasy_season2019[name]
        if score1 is not 'NaN':
            if score2 is not 'NaN':
                ave = (score1+score2)/2
            else:
                ave = 0
        else:
            ave = 0
        yeardif[name] = [(score2-score1), score1, score2, ave]
    except:
        pass
#print(yeardif)

scoredf = pd.DataFrame(yeardif, index  = ['Dif', 'Year1', 'Year2', 'Average']).T
posdf = pd.DataFrame.from_dict(dictpos, orient='index', columns = ['Position'])
namedf = pd.DataFrame.from_dict(dictname, orient='index', columns = ['Name'])
difdf = pd.concat([scoredf,posdf,namedf], axis = 1)

#print(fantasy_season2019[between['Name'].index[0]])

fig = px.scatter(difdf, 'Dif', 'Average', hover_name = 'Name',
                 color = 'Position',
                 color_discrete_sequence=('red', 'green',
                                          'black', 'blue', 'silver'),
                 title = 'Change from '+str(year1)+' to '+str(year2))

plot(fig)