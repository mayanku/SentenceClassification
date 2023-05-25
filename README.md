# SentenceClassification
This project was done as part of Information Retrieval course assignment where we had to build a model using Text GCN to classify the documents. We have used Text GCN and its extension Bert GCN to build the models
# Data Cleaning



```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/PubMed_records_for_covid-19_labelled&unlabelled.xlsx - Sheet1 (1).csv")

df = df[['Article title', 'Article keywords' ,'Article abstract', 'Contextual']].copy()

df.head()
```





  <div id="df-47734e1a-587f-4b97-ba21-7ae8a2ce5f9a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Article title</th>
      <th>Article keywords</th>
      <th>Article abstract</th>
      <th>Contextual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Assessing the impacts of COVID-19 vaccination ...</td>
      <td>Affordability;COVID-19 | SARS-CoV-2;Decision-m...</td>
      <td>The COVID-19 vaccine supply shortage in 2021 c...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Association between interleukin-10 gene polymo...</td>
      <td>COVID-19;Interleukin-10 gene polymorphisms;SAR...</td>
      <td>Polymorphisms in the interleukin-10 (IL10) gen...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Quality of Life of early-stage breast-cancer p...</td>
      <td>Breast;COVID-19;Cancer;EORTC;Oncology;Quality ...</td>
      <td>Objectives To describe the Quality of Life (QO...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The research interest, capacity and culture of...</td>
      <td>Barriers;Health Research;Innovation;Motivators...</td>
      <td>The UK National Health Service (NHS) is ideall...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Machine learning prediction for COVID-19 disea...</td>
      <td>COVID-19;Classification;Laboratory markers;Mac...</td>
      <td>Early prognostication of patients hospitalized...</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-47734e1a-587f-4b97-ba21-7ae8a2ce5f9a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-47734e1a-587f-4b97-ba21-7ae8a2ce5f9a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-47734e1a-587f-4b97-ba21-7ae8a2ce5f9a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
import nltk
nltk.download('stopwords')

```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```python
from nltk.corpus import stopwords
stop = stopwords.words('english')

print(stop)
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]



```python
import string

df['Article title'] = df['Article title'].str.lower()
df['Article keywords'] = df['Article keywords'].str.lower()
df['Article abstract'] = df['Article abstract'].str.lower()

#If accuracy is bad then try removing on relevant punctuations after assessing the data
df['Article title'] = df['Article title'].str.replace('[{}]'.format(string.punctuation.replace('-','')), '')
df['Article abstract'] = df['Article abstract'].str.replace('[{}]'.format(string.punctuation.replace('-','')), '')
df['Article keywords'] = df['Article keywords'].str.replace('[{}]'.format(string.punctuation.replace('-','')), '')

df['Article title'] = df['Article title'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
df['Article keywords'] = df['Article keywords'].apply(lambda x: ' '.join([item for item in str(x).split(';') if item not in stop]))
df['Article abstract'] = df['Article abstract'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))



```

    <ipython-input-198-cfda384d0022>:8: FutureWarning: The default value of regex will change from True to False in a future version.
      df['Article title'] = df['Article title'].str.replace('[{}]'.format(string.punctuation.replace('-','')), '')
    <ipython-input-198-cfda384d0022>:9: FutureWarning: The default value of regex will change from True to False in a future version.
      df['Article abstract'] = df['Article abstract'].str.replace('[{}]'.format(string.punctuation.replace('-','')), '')
    <ipython-input-198-cfda384d0022>:10: FutureWarning: The default value of regex will change from True to False in a future version.
      df['Article keywords'] = df['Article keywords'].str.replace('[{}]'.format(string.punctuation.replace('-','')), '')



```python
from sklearn.model_selection import train_test_split

df_labeled = df[df['Contextual'].notnull()]
df_unlabeled = df[df['Contextual'].isnull()]
df_train, df_test = train_test_split(df_labeled, test_size=0.2, shuffle=False)

labeled_index_in_train = list(df_train.index.values)
labeled_index_in_test = list(df_test.index.values)
unlabeled_index = list(df_unlabeled.index.values)

train_size = len(df_train) + len(df_test)  + len(df_unlabeled)

df_test_without_contextual = df_test[['Article title', 'Article keywords', 'Article abstract']]

df_train = pd.concat([df_train, df_test_without_contextual, df_unlabeled])

# print(df_train.head())
# print(df_test.head())

print(labeled_index_in_train)
print(labeled_index_in_test)

print(df_train.info())
print(df_test.info())
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 76, 77, 80, 81, 82, 83, 84, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144]
    [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 204, 205, 206]
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 8833 entries, 0 to 8832
    Data columns (total 4 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Article title     8833 non-null   object 
     1   Article keywords  8833 non-null   object 
     2   Article abstract  8833 non-null   object 
     3   Contextual        131 non-null    float64
    dtypes: float64(1), object(3)
    memory usage: 345.0+ KB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 57 entries, 145 to 206
    Data columns (total 4 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Article title     57 non-null     object 
     1   Article keywords  57 non-null     object 
     2   Article abstract  57 non-null     object 
     3   Contextual        57 non-null     float64
    dtypes: float64(1), object(3)
    memory usage: 2.2+ KB
    None


# Build Graph


## Build list of training documents


```python
row = []
col = []
weight = []

train_list = []

for index in df_train.index:
  train_list.append( str(index) + ':=:' + df_train['Article title'][index] + df_train['Article keywords'][index] + df_train['Article abstract'][index])  # 


```

## TF-IDF for document-word weight in graph


```python
from math import log
# doc word frequency
# TF is simple raw frequency unlike (1+log(tf)), we might want to change that if accuracy is not good

# build vocab
word_set = set()
for doc_words in train_list:
    # words = doc_words.split()
    words = doc_words.split(':=:')[1].split()
    for word in words:
        word_set.add(word)

vocab = list(word_set)
vocab_size = len(vocab)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

doc_word_freq = {}

for doc_id in range(train_size):
    doc_words = train_list[doc_id]
    # words = doc_words.split()
    words = doc_words.split(':=:')[1].split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1


word_doc_list = {}

for i in range(train_size):
    doc_words = train_list[i]
    # words = doc_words.split()
    words = doc_words.split(':=:')[1].split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

for i in range(train_size):
    doc_words = train_list[i]
    # words = doc_words.split()
    row_index = int(doc_words.split(':=:')[0])
    words = doc_words.split(':=:')[1].split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        # row.append(i)
        row.append(row_index)
        col.append(train_size + j)
        idf = log(1.0 * len(train_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

print(len(row))
print(len(col))
print(len(weight))

```

    636812
    636812
    636812


## PMI Calculation for word-word edge weight in graph


```python
#Get the train as a list and define window size
window_size = 20
windows = []

for doc_words in train_list:
    # words = doc_words.split()
    words = doc_words.split(':=:')[1].split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

# print(windows)

word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

# print(word_window_freq)

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            # print(word_i + ' ' + word_j + ' ' + str(word_i_id) + ' ' + str(word_j_id))
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            # print(word_j + ' ' + word_i + ' ' + str(word_j_id) + ' ' + str(word_i_id))
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1


# print(word_pair_count)

# pmi as weights

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    #Adjust the position of pmi weights in final adjacency matrix
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

print(len(row))
print(len(col))
print(len(weight))
```

    11435398
    11435398
    11435398


## Diagonal edge weight initialized to 1 in the adjacency matrix


```python
node_size = train_size + vocab_size

for i in range(node_size):
  row.append(i)
  col.append(i)
  weight.append(1)

```

## Adjacency Matrix : A


```python
import scipy.sparse as sp

#Train size contains all data (train + test both labeled and unlabeled)
print(len(row))
print(len(col)) 
print(len(weight))
print(node_size)

adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
print(adj)
```

    11520811
    11520811
    11520811
    85413
      (0, 0)	1.0
      (0, 9298)	3.6525279831908923
      (0, 10758)	4.768761873208821
      (0, 14637)	4.347014332503912
      (0, 14856)	3.5728212405801503
      (0, 15110)	4.137490096366964
      (0, 15357)	9.086249986745132
      (0, 20246)	4.5115390082417495
      (0, 20807)	3.502753677963433
      (0, 23773)	3.426767770985511
      (0, 27594)	2.968152788703784
      (0, 28597)	3.426767770985511
      (0, 30297)	5.397370532631196
      (0, 30500)	2.213086152532614
      (0, 31003)	6.041727549021709
      (0, 31808)	8.393102806185187
      (0, 34890)	2.559755127174342
      (0, 36022)	4.837754744695773
      (0, 36190)	7.140339837689819
      (0, 39154)	3.2993526053784246
      (0, 39197)	3.34324679893565
      (0, 39658)	0.23287031381749881
      (0, 41124)	9.086249986745132
      (0, 41339)	5.060898296009983
      (0, 42915)	5.530901925255718
      :	:
      (85411, 84936)	3.116633014507707
      (85411, 85254)	0.9595787196986144
      (85411, 85411)	1.0
      (85412, 19)	9.086249986745132
      (85412, 10532)	4.713793646043373
      (85412, 14658)	4.873122838816369
      (85412, 24178)	7.2464329508054846
      (85412, 27549)	7.612495511047619
      (85412, 27960)	9.381782124423616
      (85412, 28694)	1.6684045183383
      (85412, 36120)	6.755906531510422
      (85412, 38172)	10.991220036857717
      (85412, 39658)	0.9430502990682924
      (85412, 45880)	6.058546284231173
      (85412, 46167)	10.991220036857717
      (85412, 47911)	4.225412420531261
      (85412, 49388)	10.991220036857717
      (85412, 59542)	4.641207301429322
      (85412, 61160)	3.5575042465253186
      (85412, 62029)	6.510479929247802
      (85412, 71436)	2.72755801459223
      (85412, 76651)	9.119417859956124
      (85412, 82472)	10.991220036857717
      (85412, 82549)	3.674422721277072
      (85412, 85412)	1.0


## Build Feature Matrix: X 


```python
import math
import numpy as np

row_x = []
col_x = []
weight_x = []

# One hot vector for X as per text GCN
for i in range(node_size):
  row_x.append(i)
  col_x.append(i)
  weight_x.append(1)

x = sp.csr_matrix(
    (weight_x, (row_x, col_x)), shape=(node_size, node_size))

print(x.shape)


```

    (85413, 85413)


## Build label matrix: Y


```python
y = []
ty = []
label_list = [0,1]

for index in df_train.index:
  if index in labeled_index_in_train:
    label = int(df_train['Contextual'][index])
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
    ty.append([0,0])
  elif index in labeled_index_in_test:
    label = int(df_test['Contextual'][index])
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
    y.append([0,0])
  else:
    y.append([0,0])
    ty.append([0,0])

for i in range(vocab_size):
  y.append([0,0])
  ty.append([0,0])

y = np.array(y)
ty = np.array(ty)

# print(x)
# print(y[:200])

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

features = sp.vstack((x)).tolil()
features = sp.identity(features.shape[0])  
print(features.shape)

idx_train = labeled_index_in_train
idx_test = labeled_index_in_test

print(idx_train)
print(idx_test)

train_mask = sample_mask(idx_train, y.shape[0])
test_mask = sample_mask(idx_test, ty.shape[0])
print(train_mask)
print(test_mask)

y_train = np.zeros(y.shape)
y_test = np.zeros(ty.shape)

y_train[train_mask, :] = y[train_mask, :]
y_test[test_mask, :] = ty[test_mask, :]

print(y_train.shape)
print(y_test.shape)
```

    (85413, 85413)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 76, 77, 80, 81, 82, 83, 84, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144]
    [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 204, 205, 206]
    [ True  True  True ... False False False]
    [False False False ... False False False]
    (85413, 2)
    (85413, 2)


    <ipython-input-207-526c66212c57>:38: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      return np.array(mask, dtype=np.bool)


# Create GCN Model

## Define Flags for tensorflow


```python
import random
import tensorflow.compat.v1 as tf
from sklearn import metrics
import os
import sys
import numpy as np
import time

tf.disable_v2_behavior()

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

flags = tf.app.flags
FLAGS = flags.FLAGS
del_all_flags(FLAGS)

#Define flags
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 12, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 500, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('f', '', 'kernel')

print("TensorFlow version:", tf.__version__)
```

    TensorFlow version: 2.12.0


## Define Layer classes


```python
def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output #output
        return self.act(output)
```

## Define Model classes


```python
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    print(preds)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))

    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            featureless=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x, #
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


```

## Training the model


```python
# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)

```


```python
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
  
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


features = preprocess_features(features)
support = [preprocess_adj(adj)]

num_supports = 1
model_func = GCN

print(support)
```

    [(array([[    0,     0],
           [ 9298,     0],
           [10758,     0],
           ...,
           [82472, 85412],
           [82549, 85412],
           [85412, 85412]], dtype=int32), array([0.0089806 , 0.003025  , 0.00497513, ..., 0.07958011, 0.00289949,
           0.01441331]), (85413, 85413))]



```python
# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

print(placeholders)
```

    {'support': [SparseTensor(indices=Tensor("Placeholder_66:0", shape=(?, ?), dtype=int64), values=Tensor("Placeholder_65:0", shape=(?,), dtype=float32), dense_shape=Tensor("Placeholder_64:0", shape=(?,), dtype=int64))], 'features': SparseTensor(indices=Tensor("Placeholder_68:0", shape=(?, 2), dtype=int64), values=Tensor("Placeholder_67:0", shape=(?,), dtype=float32), dense_shape=Tensor("PlaceholderWithDefault_16:0", shape=(2,), dtype=int64)), 'labels': <tf.Tensor 'Placeholder_69:0' shape=(?, 2) dtype=float32>, 'labels_mask': <tf.Tensor 'Placeholder_70:0' shape=<unknown> dtype=int32>, 'dropout': <tf.Tensor 'PlaceholderWithDefault_17:0' shape=() dtype=float32>, 'num_features_nonzero': <tf.Tensor 'Placeholder_71:0' shape=<unknown> dtype=int32>}



```python
def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

# Create model
print(features[2][1])
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,
                     model.layers[0].embedding], feed_dict=feed_dict)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

```

    85413
    Tensor("graphconvolution_2_8/SparseTensorDenseMatMul/SparseTensorDenseMatMul:0", shape=(?, 2), dtype=float32)
    Epoch: 0001 train_loss= 0.61910 train_acc= 0.48092 time= 12.76749
    Epoch: 0002 train_loss= 0.60318 train_acc= 0.73282 time= 11.63542
    Epoch: 0003 train_loss= 0.56767 train_acc= 0.70992 time= 11.69183
    Epoch: 0004 train_loss= 0.51449 train_acc= 0.73282 time= 11.70606
    Epoch: 0005 train_loss= 0.44955 train_acc= 0.74046 time= 11.40795
    Epoch: 0006 train_loss= 0.37612 train_acc= 0.79389 time= 11.75368
    Epoch: 0007 train_loss= 0.30037 train_acc= 0.89313 time= 11.81961
    Epoch: 0008 train_loss= 0.22828 train_acc= 0.89313 time= 11.94403
    Epoch: 0009 train_loss= 0.16745 train_acc= 0.93893 time= 11.78917
    Epoch: 0010 train_loss= 0.12120 train_acc= 0.96947 time= 12.30199
    Epoch: 0011 train_loss= 0.08511 train_acc= 0.97710 time= 12.23177
    Epoch: 0012 train_loss= 0.05828 train_acc= 0.98473 time= 11.75568
    Optimization Finished!


# Test the model


```python

# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

test_pred = []
test_labels = []
print(len(test_mask))
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(test_labels, test_pred, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

```

    Test set results: cost= 0.55467 accuracy= 0.45614 time= 5.90067
    85413
    Test Precision, Recall and F1-Score...
                  precision    recall  f1-score   support
    
               0     0.5000    0.5161    0.5079        31
               1     0.4000    0.3846    0.3922        26
    
        accuracy                         0.4561        57
       macro avg     0.4500    0.4504    0.4500        57
    weighted avg     0.4544    0.4561    0.4551        57
    
    Macro average Test Precision, Recall and F1-Score...
    (0.45, 0.45037220843672454, 0.45004668534080294, None)
    Micro average Test Precision, Recall and F1-Score...
    (0.45614035087719296, 0.45614035087719296, 0.45614035087719296, None)



```python

```
