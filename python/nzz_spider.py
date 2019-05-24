#execute with scrapy runspider nzz_spider.py -o ../data/nzz.json
import scrapy
from urllib.parse import urljoin


class NZZSpider(scrapy.Spider):
    i = 0
    name = "titles"
    start_urls = [
        "https://www.nzz.ch/neueste-artikel/"
    ]

    def parse(self, response):
        self.i+=1
        for title in response.xpath("//div[@class='teaser__content']"):
            yield{
                'title': title.xpath("./a/h2/span/text()").extract_first()
            }
        
        next_page = urljoin('https://www.nzz.ch/neueste-artikel',
                            'https://www.nzz.ch/neueste-artikel/?page='+str(self.i))
        print("NÄCHSTE SEITE", next_page)
        if next_page is not None and self.i<3:
            next_page_link = response.urljoin(next_page)
            yield scrapy.Request(url=next_page_link, callback=self.parse, dont_filter=True)



