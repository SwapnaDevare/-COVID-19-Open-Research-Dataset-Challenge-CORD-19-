#!/usr/bin/env python
# coding: utf-8

# Here we are going to use VADER Polarity Score to generate our sentimental analysis on text. VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labeled according to their semantic orientation as either positive or negative. VADER not only tells about the Positivity and Negativity score but also tells us about how positive or negative a sentiment is.

# In[1]:


import pandas as pd
import numpy as np
from zipfile import ZipFile


# In[2]:


df = pd.read_csv("C:/Users/admin/Downloads/metadata.csv", nrows=100)


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


#As there are 100 out of 100 null objects we can drop those columns as it does not have any impact on our analysis
df.drop(['mag_id','who_covidence_id','arxiv_id','s2_id'],axis =1,inplace = True)


# In[7]:


df.isnull().sum()


# In[8]:


df.head(2)


# In[9]:


#Finding number of articles published by each journal
df.groupby(['journal']).size().groupby(level=0).max()


# We find that the Nuleic Acids Res has got the highest number of publications

# In[10]:


df.groupby(['license']).size().groupby(level=0).max()


# Most publications i.e 46 were published under the cc-by license type compared to the cc0 type with only 2 publications

# In[11]:


df.groupby(['publish_time']).size().groupby(level = 0).max()


# Extracting the title and abstract column for sentiment analysis

# In[12]:


metadata = df[['title','abstract']]
metadata.head()


# In[13]:


metadata.isnull().sum()


# It shows there are 8 null objects in abstract column

# In[14]:


metadata.reset_index(drop = True, inplace = True)
metadata.info()


# In[15]:


metadata['text'] = metadata['title'] + metadata['abstract']
metadata


# In[16]:


metadata['text'] = metadata['text'].astype(str).str.lower()
metadata.head()


# In[17]:


from nltk.tokenize import RegexpTokenizer

regexp = RegexpTokenizer('\w+')

metadata['text_token'] = metadata['text'].apply(regexp.tokenize)

metadata.head()


# In[18]:


from nltk.tokenize import RegexpTokenizer

regexp = RegexpTokenizer('\w+')

metadata['text_token'] = metadata['text'].apply(regexp.tokenize)

metadata.head()


# In[19]:


import nltk
from nltk.corpus import stopwords

# Make a list of english stopwords
stopwords = nltk.corpus.stopwords.words("english")


# In[20]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')


# Remove stopwords

# In[21]:


metadata['text_token'] = metadata['text_token'].apply(lambda x: [item for item in x if item not in stopwords])
metadata.head()


# In[22]:


metadata['text_string'] = metadata['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))


# In[23]:


metadata[['text','text_token','text_string']].head()


# In[24]:


#Create a list of all words

all_words = ' '.join([word for word in metadata['text_string']])


# In[25]:


#Tokenize all_words
tokenized_words = nltk.tokenize.word_tokenize(all_words)


# In[26]:


from nltk.probability import FreqDist

fdist = FreqDist(tokenized_words)
fdist


# In[27]:


metadata['text_string_fdist'] = metadata['text_token'].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 1 ]))


# In[28]:


fdist.most_common


# In[29]:


fdist.tabulate(5)


# In[30]:


# Obtain top 10 words
top_10 = fdist.most_common(10)

# Create pandas series to make plotting easier
fdist = pd.Series(dict(top_10))


# In[31]:


import seaborn as sns
sns.set_theme(style="ticks")

sns.barplot(y=fdist.index, x=fdist.values, color='blue');


# Search Specific words

# In[32]:


# Show frequency of a specific word
fdist["viral"]


# In[33]:


pip install wordcloud


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(width=600, 
                     height=400, 
                     random_state=2, 
                     max_font_size=100).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');


# In[35]:


#Different style:
import numpy as np

x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

wc = WordCloud(background_color="white", repeat=True, mask=mask)
wc.generate(all_words)

plt.axis("off")
plt.imshow(wc, interpolation="bilinear");


# ### Sentiment analysis 

# ## VADER lexicon
# NLTK provides a simple rule-based model for general sentiment analysis called VADER, which stands for “Valence Aware Dictionary and Sentiment Reasoner” (Hutto & Gilbert, 2014).

# ## Sentiment Intensity Analyzer
# Initialize an object of SentimentIntensityAnalyzer with name “analyzer”:

# In[36]:


import nltk
nltk.downloader.download('vader_lexicon')


# In[37]:


from nltk.sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


# ## Polarity scores
# Use the polarity_scores method:

# In[38]:


metadata['polarity'] = metadata['text_string_fdist'].apply(lambda x: analyzer.polarity_scores(x))
metadata.tail(3)


# ## Transform data
# Change data structure

# In[39]:


metadata1 = pd.concat([metadata['polarity'].apply(pd.Series)], axis=1)
metadata1.head(3)


# In[40]:


# Create new variable with sentiment "neutral," "positive" and "negative"
metadata1['sentiment'] = metadata1['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
metadata1.head()


# ## Analyze data
# Title with highest positive sentiment

# In[41]:


metadata1.loc[metadata1['compound'].idxmax()].values


# Title with highest negative sentiment

# In[42]:


metadata1.loc[metadata1['compound'].idxmin()].values


# ## Visualize data

# In[43]:


# Number of tweets 
sns.countplot(y='sentiment', 
             data=metadata1, 
             palette=['#b2d8d8',"#008080", '#db3d13']
             );


# In[44]:


# Boxplot
sns.boxplot(y='compound', 
            x='sentiment',
            palette=['#b2d8d8',"#008080", '#db3d13'], 
            data=metadata1);


# In[45]:


# Lineplot
g = sns.lineplot(x=df['publish_time'], y=metadata1['compound'])


g.set(title='Sentiment of Titles')
g.set(xlabel="Time")
g.set(ylabel="Sentiment")


g.axhline(0, ls='--', c = 'grey');


# In[46]:


metadata1.info()


# In[ ]:




