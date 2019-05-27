from bs4 import BeautifulSoup
import requests

#understand xpath http://zvon.org/comp/r/tut-XPath_1.html

def crawl_postillion(url):
    result = []
    req = requests.get(url)
    i=0
    while(req is not None and i<2):
      soup = BeautifulSoup(req.text, "lxml")
      containers=soup.find_all("div",{"class":"post hentry"}) 
      for contain in containers:
        result.append(contain.h3.a.contents)
      #get the url to older pages
      
      i+=1
    print(result)




crawl_postillion("https://www.der-postillon.com/")

