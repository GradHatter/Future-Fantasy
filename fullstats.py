# -*- coding: utf-8 -*-

import pandas as pd
from bs4 import BeautifulSoup
import requests
import lxml
from selenium import webdriver
from time import sleep

url = 'http://www.pro-football-reference.com'

years_to_scrape = [i for i in range(2022,2023)]

def parse(url):
    response = webdriver.Chrome()
    response.get(url)
    sleep(4)
    sourceCode=response.page_source
    return  sourceCode

def get_players_table(year):
    
    # grab fantasy players
    r = requests.get(url + '/years/' + str(year) + '/fantasy.htm')
    soup = BeautifulSoup(r.content, 'lxml')
    #soup = BeautifulSoup(parse(url + '/years/{}/fantasy.htm'.format(year)), 'lxml')
    parsed_table = soup.find('table', id = 'fantasy')     
    table = parsed_table.find('tbody')
    sleep(4)
    
    return table

for year in years_to_scrape:
    maxp = 750
    
    #table = get_players_table(year)
    
    # grab fantasy players
    r = requests.get(url + '/years/' + str(year) + '/fantasy.htm')
    soup = BeautifulSoup(r.content, 'lxml')
    #soup = BeautifulSoup(parse(url + '/years/{}/fantasy.htm'.format(year)), 'lxml')
    parsed_table = soup.find('table', id = 'fantasy')      
    table = parsed_table.find('tbody')
    sleep(4)
    
    df = []
    
    # first row is col headers
    n = 0
    for i,row in enumerate(table.find_all('tr')):
        if i >= maxp: 
            print('\nComplete.')
            break
        elif i % 10 == 0: 
            print(i, end=' ')
        else:
            pass
        
        if i == 29+31*n:
            n+=1
            pass
        else:
            dat = row.find('td', attrs={'data-stat': 'player'})
            playerid = dat.get('data-append-csv')
            name = dat.a.get_text()
            stub = dat.a.get('href')
            stub = stub[:-4] + '/gamelog/{}'.format(year)
            pos = row.find('td', attrs={'data-stat': 'fantasy_pos'}).get_text()
            sleep(4)
            
            print(
                  #url+stub, '\n',
                  name, '\n',
                  #playerid, '\n',
                  #pos, '\n',
                  )
            
            player_r = requests.get(url + stub)
            playersoup = BeautifulSoup(player_r.content, 'lxml')
            #playersoup = BeautifulSoup(parse(url + stub), 'lxml')
            #print(playersoup)
            #player_table = playersoup.find('table', id = 'stats')#.find('tbody')
            # grab this players stats
            tdf = pd.read_html(str(playersoup), attrs = {'id' : 'stats'})[0]
            
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
            #print(df)
        

    df = pd.concat(df, axis = 0, ignore_index = True)
    df = df.fillna(0)
    
    csv_name = 'player{}.csv'.format(str(year))

    df.to_csv(csv_name)
