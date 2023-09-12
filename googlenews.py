# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:14:09 2022

@author: oleg.kazanskyi
"""

import json
import numpy as np
import math
from datetime import datetime

googlenews = GoogleNews(lang='en', region='US', start='02/01/2020',end='02/28/2022', encode='utf-8')
results = googlenews.results()
googlenews.get_news('3M company')
print(googlenews.total_count())
texts = googlenews.get_texts()
googlenews.get_news('3M company')

l = []

for i in results:
    l.append(i["datetime"])

l = list(filter(None, l))
l = list(filter(lambda i:not(type(i) is not datetime), l))
l.sort()
