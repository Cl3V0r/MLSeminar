import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path



contents = Path('../data/mixed_news/fake_news_titles.csv').read_text()
wordcloud = WordCloud(background_color='white',
                      width=1920,
                      height=1080
                      ).generate(contents)
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig("../build/plots/fake_news_titles_wordcloud.pdf")
