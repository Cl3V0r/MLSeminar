import scrapy

class PostillonSpider(scrapy.Spider):
    name="titles"

    start_urls = [
        "https://www.der-postillon.com/"
    ]

    def parse(self, response):
        for title in response.xpath("//h3[@class='post-title entry-title']"):
            yield{
                'title': title.xpath(".//a").extract_first()
            }
        next_page = response.xpath("//div[@class='blog-pager']/span/a/@href").extract_first()
        if next_page is not None:
            next_page_link = response.urljoin(next_page)
            yield scrapy.Request(url=next_page_link , callback=self.parse)   


#example for a part which has to be extracted from the .html file
#<h3 class = 'post-title entry-title' itemprop = 'name' >
#<a href = 'https://www.der-postillon.com/2019/05/hardware-sicher-entfernen.html' >
#USB-Stick gezogen, ohne ihn sicher zu entfernen: Mann geht in Flammen auf
#</a >
#</h3 >

#or the complete div
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
#<div class = "post-body entry-content" id = "post-body-1486585981545012902" itemprop = "articleBody" >
#<div class = "separator" style = "clear: both; text-align: center;" >
#<a href = "https://www.der-postillon.com/2019/05/cdu-vhs-rezo.html" imageanchor = "1" style = "clear: left; float: left; margin-bottom: 1em; margin-right: 1em;" > <img data-original-height = "1067" data-original-width = "1600" src = "https://1.bp.blogspot.com/-jfejfeOBQ4U/XOUp94H6QdI/AAAAAAAAzpA/PCroh-ZTzzUMwzccT01yQ4fN2BnaUOqjwCLcBGAs/s1600/Rezo-VHS.jpg" width = "548" border = "0" > </a > </div >
#Berlin(dpo) - Das konnte die CDU nicht auf sich sitzen lassen: < a href = "https://www.youtube.com/watch?v=4Y1lZQsyuSQ" target = "_blank" > Nach dem mit über 3 Millionen Views viral gegangenen Kritik-Video von YouTuber Rezo < /a > hat die Partei heute ihr Antwortvideo präsentiert. Der Film mit dem Titel "CDU zerstört Rezo." ist ab sofort auf VHS erhältlich und kann in jeder gut sortierten Videothek ausgeliehen werden. < br > <space > </space >
#<a class = "more-link" href = "https://www.der-postillon.com/2019/05/cdu-vhs-rezo.html#more" title = "Jetzt auf VHS-Kassette! CDU veröffentlicht Antwort auf YouTuber Rezo" > mehr... < /a >
#<div style = "clear: both;" > </div >
#</div >
#<div class = "post-footer" >
#<div class = "post-footer-line post-footer-line-1" >
#<span class = "post-icons" >
#<span class = "item-control blog-admin pid-2135237929" >
#<a href = "https://www.blogger.com/post-edit.g?blogID=746298260979647434&amp;postID=1486585981545012902&amp;from=pencil" title = "Post bearbeiten" >
#<img alt = "" class = "icon-action" src = "//img2.blogblog.com/img/icon18_edit_allbkg.gif" width = "18" height = "18" >
#</a >
#</span >
#</span >
#</div >
#<div class = "post-footer-line post-footer-line-2" > </div >
#<div class = "post-footer-line post-footer-line-3" >
#<span class = "post-location" >
#</span >
#</div >
#</div >
#</div >



#the corresponding xpath for the text is: //h3/a/text()  ?
#to get to the next page use href link from 
#<div class = "blog-pager" id = "blog-pager" >
#<span id = "blog-pager-older-link" >
#<a class = "blog-pager-older-link" href = "https://www.der-postillon.com/search?updated-max=2019-05-21T13:15:00%2B02:00&amp;max-results=9" id = "Blog1_blog-pager-older-link" title = "Ältere Posts" >‹ Ältere Posts < /a >
#</span >
#</div >
