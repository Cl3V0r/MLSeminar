import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
import numpy as np
from PIL import Image
from PIL import ImageFilter

def plotWordcloud(s,t):
    if(t!=""):
       mask = np.array(Image.open('../data/pictures/'+t))
    else:
        mask=None

    contents = Path('../build/preprocessed/'+s+".csv").read_text()
    wordcloud = WordCloud(background_color='black',
                      width=1920,
                      height=1080,
                      mask=mask
                      ).generate(contents)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.savefig("../build/plots/"+s+"_wordcloud.pdf")


plotWordcloud("fake_news_titles_stem","trump_silhouette.png")
plotWordcloud("fake_news_titles_lem", "trump_silhouette.png")
plotWordcloud("real_news_titles_lem","")

