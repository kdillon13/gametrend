#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create steamspy.csv

Makes individual API calls to Steam.com for each ID in idlist.csv.
API calls write game feature data for each ID into games.json.
Steam limits API calls, so the script pauses for 5min 10sec after
every 150 API calls.

Code modified from
https://github.com/CraigKelly/steam-data

"""

import csv
import json
import time
import requests
import logging


def parse_id(i):
    """Since we deal with both strings and ints, force appid to be correct."""
    try:
        return int(str(i).strip())
    except:
        return None
def id_reader():
    """Read the previous created idlist2.csv."""
    with open("idlist2.csv") as basefile:
        reader = csv.DictReader(basefile)
        for row in reader:
            yield parse_id(row['ID']), row['Name']

URL = "http://steamspy.com/api.php"

with open("steamspy.csv", "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["appid", 
                "name", 
                "developer", 
                "positive", 
                "negative", 
                "userscore", 
                "owners",
                "average_forever",
                "average_2weeks",
                "median_forever",
                "median_2weeks",
                "price",
                "initialprice"])
    
    myparams = {
        'request': 'appdetails'
    }
        
    for appid, appname in id_reader():
        myparams=dict(myparams)
        myparams["appid"]=appid
#        print(myparams)
#        print(URL)
        resp_data = requests.get(URL, params=myparams)
#        print(resp_data)
        x = resp_data.json()
#        print(x)
        csv_writer.writerow([x["appid"], 
                    x["name"], 
                    x["developer"], 
                    x["positive"], 
                    x["negative"],
                    x["userscore"],
                    x["owners"],
                    x["average_forever"],
                    x["average_2weeks"],
                    x["median_forever"],
                    x["median_2weeks"],
                    x["price"],
                    x["initialprice"]])












