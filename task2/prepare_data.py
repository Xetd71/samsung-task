
# coding: utf-8

# # Подготавливаем данные и сохраняем их в data/data.cv

# In[1]:


import os
import re
import string
import numpy as np
import pandas as pd
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# In[2]:


sent_num = 3
gegel_files = [os.path.join("data", "Гегель Фридрих. Наука логики.txt")]
gogol_files = [os.path.join("data", "Гоголь Николай. Мертвые души.txt")]


# ## Разделим произведения Гоголя и Гегеля на куски и уберем ненужную информацию

# In[3]:


stop_words = set(stopwords.words("russian"))
stemmer = SnowballStemmer("russian")

def read_split_data(files_paths):
    # читаем тексты
    text = ""
    for file_path in files_paths:
        with open(file_path) as f:
            text += os.linesep + f.read()
            
    # делим тексты на предложения        
    sentences = sent_tokenize(text)
    
    data, fragment = [], []
    # объединяем в группу по sent_num предложений
    for i in np.arange(0, len(sentences), sent_num):
        s = " ".join(sentences[i:i + sent_num])
        fragment.append(s)
        # делим на слова
        s = word_tokenize(s)
        # приводим к нижнему регистру, убираем пунктуацию
        s = [w.lower().translate(str.maketrans('', '', string.punctuation)) for w in s]
        # фильтруем слова не из алфавита
        s = [w for w in s if w.isalpha()]
        # фильтруем стоп слова
        s = [w for w in s if not w in stop_words]
        # преобразуем в коренные слова
        s = [stemmer.stem(w) for w in s]
        data.append(" ".join(s))
    return fragment, data


# In[4]:


gegel_fragment, gegel_data = read_split_data(gegel_files)
gogol_fragment, gogol_data = read_split_data(gogol_files)


# ## Сохраним данные в файл data/data.csv

# In[5]:


fragment = np.array(gegel_fragment + gogol_fragment, dtype="str")
X = np.array(gegel_data +  gogol_data, dtype="str")
y = np.array([0]*len(gegel_data) + [1]*len(gogol_data), dtype="int")


# In[6]:


df = pd.DataFrame(data={"Fragment": fragment, "X": X, "y": y})
df.head()


# In[7]:


df.to_csv("data/data.csv", sep=',', encoding="cp1251", index=False)

