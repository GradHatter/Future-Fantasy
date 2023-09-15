# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 22:02:41 2020

@author: jkora
https://www.pro-football-reference.com/players/J/JoneJu02/gamelog/2019

@author: https://stmorse.github.io/journal/pfr-scrape-python.html
"""

import pandas as pd
from bs4 import BeautifulSoup
import requests
import lxml

url = 'https://www.pro-football-reference.com'

years_to_scrape = [i for i in range(2020,2023)]

for year in years_to_scrape:
    maxp = 2000
    
    # grab fantasy players
    r = requests.get(url + '/years/' + str(year) + '/fantasy.htm')
    soup = BeautifulSoup(r.content, 'html.parser')
    parsed_table = soup.find_all('table')[0]  
        
    df = []
    
    # first row is col headers
    for i,row in enumerate(parsed_table.find_all('tr')[2:]):
        if i % 10 == 0: print(i, end=' ')
        if i >= maxp: 
            print('\nComplete.')
            break
            
        try:
            dat = row.find('td', attrs={'data-stat': 'player'})
            name = dat.a.get_text()
            stub = dat.a.get('href')
            stub = stub[:-4] + '/gamelog/' + str(year)
            pos = row.find('td', attrs={'data-stat': 'fantasy_pos'}).get_text()
            
            # grab this players stats
            tdf = pd.read_html(url + stub)[0]
                    
            colnames = []
            ind = 0
            # make new column names
            while ind < len(tdf.columns.get_level_values(-2)):
                if "Unnamed:" in tdf.columns.get_level_values(-2)[ind]:
                    colnames.append(tdf.columns.get_level_values(-1)[ind])
                else:
                    col = tdf.columns.get_level_values(-2)[ind] + ' ' +\
                        tdf.columns.get_level_values(-1)[ind]
                    colnames.append(col)
                ind += 1
                  
            # get rid of MultiIndex, just keep new column names
            tdf.columns = colnames
            
            # fix the away/home column
            tdf = tdf.rename(columns={'Unnamed: 6_level_1': 'Away'})
            tdf['Away'] = [1 if r=='@' else 0 for r in tdf['Away']]
            
            # fix game started column
            tdf['GS'] = [1 if r=='*' else 0 for r in tdf['GS']]
    
            # drop last 4 columns
            #tdf = tdf.iloc[:,:-4]
            
            # drop "Total" row
            #tdf = tdf.query('Date != "Total"')
            
            # drop last row
            tdf.drop(tdf.tail(1).index,inplace=True)
            
            # add other info
            tdf['Name'] = name
            tdf['Position'] = pos
            tdf['Season'] = year
            tdf['Player ID'] = stub[11:-13]
            
            df.append(tdf)
        except:
            pass

    df = pd.concat(df, axis = 0, ignore_index = True)
    df = df.fillna(0)
    
    csv_name = 'player{}.csv'.format(str(year))

    df.to_csv(csv_name)
