# SentenceClassification
This project was done as part of Information Retrieval course assignment where we had to build a model using Text GCN to classify the documents. We have used Text GCN and its extension Bert GCN to build the models

**Problem Statement:**

- The objective of the assignment is to identify the medical research proposals which involve studying the SARS-CoV-2 model.

- Develop a machine learning model which can classify the research proposals on the basis of submitted title, keywords and abstract information (AKT)

**Steps Followed**

**Cleaning the Data and Preparing the Graph**

- Upon examining our data, we identified the presence of additional columns that needed to be cleaned. To address this, we created a new dataframe that includes the Title, Article Keywords, Article Abstract, and Contextual columns. The Contextual column indicates whether the record is related to COVID or not.

![](RackMultipart20230526-1-ow69lj_html_cace3366a4c4def7.png)

- To clean the data, we initially converted all columns to lowercase. Next, we removed stopwords from each column using the nltk package and its stopwords file. We also eliminated punctuation from the data, except for the hyphen ("-"). Through various permutations, we discovered that the hyphen is important, particularly in terms like "COVID-19," so we decided to retain it.

![](RackMultipart20230526-1-ow69lj_html_25962f59027fb4ec.png)

- Train-Test Split: We performed various splits on our data. Initially, we used an 80:20 split without shuffling. During this step, we also created our test data, which excluded the Contextual information. Consequently, our training data consisted of the following: df\_train = pd.concat([df\_train, df\_test\_without\_contextual, df\_unlabeled]).

![](RackMultipart20230526-1-ow69lj_html_bc02fb5eb1eec94d.png)

- Considering that the majority of our data was unlabeled, with limited labeled data, we opted to employ the TextGCN model. The input for this model is a graph, and to construct the graph, we utilized a matrix of dimensions (n+m)\*(n+m), where n represents the number of documents and m denotes the total number of unique words. To populate the matrix, we utilized tf-idf values for relationships between words and documents. For relationships between words, we utilized PMI scores. Additionally, each word and document included a self-loop, resulting in diagonal entries with a value of 1. To represent this matrix, we employed a sparse matrix as follows:

- adj = sp.csr\_matrix((weight, (row, col)), shape=(node\_size, node\_size)) where node\_size is the sum of all documents and unique words.

- With this, our graph-building process is complete.

![](RackMultipart20230526-1-ow69lj_html_61da375f8c7dbe40.png)

**Model Creation**

Our Initial Feature matrix consists of one hot vector of size equal to the number of nodes in our graph.

![](RackMultipart20230526-1-ow69lj_html_4ac4c695e5a5fa8e.png)

**Label Matrix:**

- The label matrix consists of a y vector, where:
  - For label=1, the label is [0 1].
  - For label=0, the label is [1 0].
  - For unlabeled and test data, the label is [0 0].

**Note:-** All the classes are taken from git reference shared in the paper for TextGCN and BertGCN

**Architecture**

![](RackMultipart20230526-1-ow69lj_html_9052e98cba592f30.png)

- To enhance this accuracy, we recognized the need to improve our initial feature matrix. As a solution, we incorporated embeddings into the initial feature matrix. We employed BERT to generate embeddings for both documents and words and stored them in a file. However, due to the large size of the files, we faced difficulties processing them.

- Subsequently, we implemented the BERT GCN Model. This model utilizes the BERT embeddings as the initial feature matrix. Each sentence and word obtain their embeddings from BERT, and the GCN model is then applied to this feature matrix.

**BERT GCN**

- BertGCN constructs a heterogeneous graph over the dataset and represents documents as nodes using BERT representations. By jointly training the BERT and GCN modules within BertGCN, the proposed model can leverage the advantages of both worlds: large-scale pretraining, which takes advantage of the massive amount of raw data, and transductive learning, which jointly learns representations for both training data and unlabeled test data by propagating label influence through graph convolution.

- We are using Roberta based pre trained BERT GCN Model. The prediction will be done by linear interpolation between pretrained Bert model and GCN model.

- RoBERTa Encoding: A pre-trained RoBERTa model is used to encode the tokens and generate contextualized word embeddings. RoBERTa incorporates advanced techniques such as dynamic masking, larger training corpus, and longer training duration to enhance language understanding.

- The X feature matrix will be populated with the embeddings of the pretrained model and the size of X will become =\> (n+m) X 128

- This model works in 3 steps:
  - Update the document embeddings of documents from a pretrained model.Initially all the features will be 0. After this function call only the features of documents will be updated keeping the word features as all 0's.

![](RackMultipart20230526-1-ow69lj_html_fdb6847312e4ece0.png)

  - Train the model where the loss is computed only on labeled train data. For this purpose a **mask variable** is used which is a boolean array. It will only consider the datapoints with value 1.

![](RackMultipart20230526-1-ow69lj_html_898e240965aab0bb.png)

  - Last step after the training is to test the accuracy of the model against the test data.

![](RackMultipart20230526-1-ow69lj_html_99b329fa7d835fdd.png)

**Result**

- By utilizing the TextGCN model, we achieved an accuracy of 58% on 80-20 split of labeled data.

- By utilizing the BertGCN model, we achieved an accuracy of 71% on 80-20 split of labeled data.

**Conclusion**

- Performance Comparison: The BertGCN model outperformed the TextGCN model in accuracy. With an 80-20 split of labeled data, the BertGCN model achieved an accuracy of 71%, while the TextGCN model achieved an accuracy of 58%. This indicates that incorporating the BERT embeddings into the GCN model resulted in improved predictive capabilities.

- Importance of pre trained Models: Leveraging pre-trained models like BERT and Roberta proved beneficial. These models, trained on large amounts of text data, provided valuable language understanding and feature extraction capabilities, enhancing the performance of the downstream GCN model.

- Future Directions: Further improvements can be explored by fine-tuning the BERT component or experimenting with different graph construction techniques. Additionally, expanding the labeled dataset or incorporating semi-supervised learning approaches can potentially enhance the accuracy and generalization of the model.

- Overall, the project highlights the effectiveness of combining BERT embeddings with a Graph Convolutional Network, showcasing improved performance in text classification tasks compared to using GCNs alone. The achieved accuracy of 71% demonstrates the potential impact of this approach in practical applications.

**Future Work**

- We can provide the word embeddings to the BertGCN model which are right now consider as 0 using techniques like word2vec which might enhance the accuracy of the model.

**References**

- [https://github.com/ZeroRin/BertGCN](https://github.com/ZeroRin/BertGCN)
- https://github.com/yao8839836/text\_gcn
