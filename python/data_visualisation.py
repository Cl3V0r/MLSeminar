import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
import numpy as np
from PIL import Image
from PIL import ImageFilter

trump_mask = np.array(Image.open('../data/pictures/trump_silhouette.png'))
contents = Path('../build/preprocessed/fake_news_titles.csv').read_text()
wordcloud = WordCloud(background_color='black',
                      width=1920,
                      height=1080,
                      mask = trump_mask
                      ).generate(contents)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.savefig("../build/plots/fake_news_titles_wordcloud.pdf")

contents = Path('../build/preprocessed/real_news_titles.csv').read_text()
wordcloud = WordCloud(background_color='white',
                      width=1920,
                      height=1080
                      ).generate(contents)
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig("../build/plots/real_news_titles_wordcloud.pdf")
