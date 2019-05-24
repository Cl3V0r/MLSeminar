#execute with scrapy runspider postillon_spider.py -o ../data/postillon.json
import scrapy

class PostillonSpider(scrapy.Spider):
    name="titles"

    start_urls = [
        "https://www.der-postillon.com/"
    ]

    def parse(self, response):
        for title in response.xpath("//div[@class='post hentry']"):
            yield{
                'title': title.xpath("./h3/a/text()").extract_first()
            }
        next_page = response.xpath("//div[@class='blog-pager']/span[@id='blog-pager-older-link']/a/@href").extract_first()
        print("NÄCHSTE SEITE",next_page)
        if next_page is not None:
            next_page_link = response.urljoin(next_page)
            yield scrapy.Request(url=next_page_link, callback=self.parse, dont_filter=True)
