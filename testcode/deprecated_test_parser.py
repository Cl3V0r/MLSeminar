from bs4 import BeautifulSoup
import requests
import json


def crawl_kompas(url):

    result = []
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "lxml")

    #find paging page
    paging = soup.find_all("div", {'class': 'paging clearfix'})
    paging_link = paging[0].find_all('a', {'class': 'paging__link'})
    last_page = int([item.get('href').split('/')[-1]
                     for item in paging_link][-1])

    #looping through paging
    for i in range(1, last_page):
        print(url+str(i))

        #find article link
        req = requests.get(url+str(i))
        soup = BeautifulSoup(req.text, "lxml")
        news_links = soup.find_all("div", {'class': 'article__list clearfix'})

        #looping through article link
        for idx, news in enumerate(news_links):
            news_dict = {}

            #find news title
            title_news = news.find('a', {'class': 'article__link'}).text

            #find urll news
            url_news = news.find('a', {'class': 'article__link'}).get('href')

            #find news content in url
            req_news = requests.get(url_news)
            soup_news = BeautifulSoup(req_news.text, "lxml")

            #find news content
            news_content = soup_news.find("div", {'class': 'read__content'})

            #find paragraph in news content
            p = news_content.find_all('p')
            content = ' '.join(item .text for item in p)
            #news_content = content.encode('utf8', 'replace')#DEPRECATED PYTHON 2 STYLE

            #wrap in dictionary
            news_dict['id'] = idx
            news_dict['url'] = url_news
            news_dict['title'] = title_news
            news_dict['content'] = news_content
            result.append(news_dict)

    return result


url = 'http://indeks.kompas.com/news/2017-08-04/'
crawl = crawl_kompas(url)
with open("../data/html/test/kompas.json", "w") as f:
    json.dump(crawl, f)
