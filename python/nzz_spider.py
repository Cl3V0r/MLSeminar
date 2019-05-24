#execute with scrapy runspider nzz_spider.py -o ../data/nzz.json
import scrapy


class NZZSpider(scrapy.Spider):
    name = "titles"
    start_urls = [
        "https://www.nzz.ch/neueste-artikel/"
    ]

    def parse(self, response):
        for title in response.xpath("//div[@class='teaser__content']"):
            yield{
                'title': title.xpath("./a/h2/span/text()").extract_first()
            }
        next_page = response.xpath(
            "//div[@class='pagination']/a[@class='pagination__next']/@href").extract_first()
        print("NÄCHSTE SEITE", next_page)
        if next_page is not None:
            next_page_link = response.urljoin(next_page)
            yield scrapy.Request(url=next_page_link, callback=self.parse, dont_filter=True)



