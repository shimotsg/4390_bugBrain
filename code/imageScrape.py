# scrape google image python - aaronsherwood https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57
# modified by gs

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib
import argparse
import wget

gpath = '/usr/local/bin/'

searchterm = 'allinurl: bugguide.net images raw' # will also be the name of the folder
url = "https://www.google.com/search?q="+searchterm+"&source=lnms&tbm=isch&sa=X&ved=0ahUKEwitjvHO_vneAhVMVK0KHYsjD08Q_AUIDigB&biw=1680&bih=950"
# NEED TO DOWNLOAD CHROMEDRIVER, insert path to chromedriver inside parentheses in following line
browser = webdriver.Firefox(gpath)
browser.get(url)
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
counter = 0
succounter = 0

if not os.path.exists(searchterm):
    os.mkdir(searchterm)

for _ in range(1000):
    browser.execute_script("window.scrollBy(0,100000)")

for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
    counter = counter + 1
    print ("Total Count:", counter)
    print ("Succsessful Count:", succounter)
    print ("URL:",json.loads(x.get_attribute('innerHTML'))["ou"])
    dl_this = json.loads(x.get_attribute('innerHTML'))["ou"]

    # img = json.loads(x.get_attribute('innerHTML'))["ou"]
    # imgtype = json.loads(x.get_attribute('innerHTML'))["ity"]
    try:
        filename = wget.download(dl_this)
        # req = urllib.Request(img, headers={'User-Agent': header})
        # raw_img = urllib.urlopen(req).read()
        # File = open(os.path.join(searchterm , 'img' + str(counter) + "." + imgtype), "wb")
        # File.write(raw_img)
        # File.close()
        succounter = succounter + 1
    except:
            print ("can't get img")

print (succounter, "pictures succesfully downloaded")
browser.close()

#https://www.google.com/search?q=allinurl:+bugguide.net+images+raw&safe=off&source=lnms&tbm=isch&sa=X&ved=0ahUKEwitjvHO_vneAhVMVK0KHYsjD08Q_AUIDigB&biw=1680&bih=950
#https://www.google.com/search?q=allinurl:+bugguide.net+images+raw&source=lnms&tbm=isch&sa=X&ved=0ahUKEwitjvHO_vneAhVMVK0KHYsjD08Q_AUIDigB&biw=1680&bih=950

