import requests
import sys, time
from requests_html import HTMLSession
import datetime

url = 'http://www.bjhd.gov.cn/zfxxgk/auto4496_51791/'
session = HTMLSession()
order = 2
while True:
    r = session.get(url)
    list_url = url + r.html.find('iframe#DataList', first = True).attrs['src'][2:]
    list_r = session.get(list_url)
    table = list_r.html.find('table.listTab.mt20', first = True)
    tds = table.find('td.mc')
    tds_date = table.find('td.rq')
    tds_date_order = table.find('td.xh')
    pat = '海淀区2021年引进非北京生源毕业生名单公示'
    msg = None
    for i, t in enumerate(tds):
        if pat in t.text \
        and tds_date[i].text == datetime.datetime.now().strftime('%Y-%m-%d')\
        and str(order) == tds_date_order[i].text:
            msg = 'public the %d-th!' % order

    if msg:
        print(msg)
        break
    else:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' not public yet')
    
    time.sleep(5*60)
        
