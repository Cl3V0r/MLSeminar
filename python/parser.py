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


#<div class = "post hentry" itemprop = "blogPost" itemscope = "itemscope" itemtype = "http://schema.org/BlogPosting" >
#<meta content = "https://1.bp.blogspot.com/-jfejfeOBQ4U/XOUp94H6QdI/AAAAAAAAzpA/PCroh-ZTzzUMwzccT01yQ4fN2BnaUOqjwCLcBGAs/s1600/Rezo-VHS.jpg" itemprop = "image" >
#<a name = "1486585981545012902" > </a >
#<h3 class = "post-title entry-title" itemprop = "name" >
#<a href = "https://www.der-postillon.com/2019/05/cdu-vhs-rezo.html" >
#Jetzt auf VHS-Kassette! CDU veröffentlicht Antwort auf YouTuber Rezo
#</a >
#</h3 >
#<div class = "post-header" >
#<div class = "post-header-line-1" > </div >
#</div >
