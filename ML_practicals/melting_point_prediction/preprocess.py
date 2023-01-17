import csv
import os
import requests
from bs4 import BeautifulSoup

drugID = []
with open("raw_data/rough.csv", 'r') as csvf:
    for i in csv.reader(csvf):
        drugID.append(i[0])
drugID[100:200]

