# CS-E4890 – Jigsaw Unintended Bias in Toxicity Classification

- [CS-E4890 – Jigsaw Unintended Bias in Toxicity Classification](#cs-e4890-%E2%80%93-jigsaw-unintended-bias-in-toxicity-classification)
  - [Abstract](#abstract)
  - [Introduction](#introduction)
    - [Scope of the project](#scope-of-the-project)
  - [Imports and Utilities](#imports-and-utilities)
  - [Analysing the datasets](#analysing-the-datasets)
    - [Cleaning up the data](#cleaning-up-the-data)
    - [Visualizations of the data](#visualizations-of-the-data)
  - [Methods](#methods)
    - [Preprocess the data](#preprocess-the-data)
    - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
    - [Long short-term memory (LSTM)](#long-short-term-memory-lstm)
      - [The network](#the-network)
    - [Convolutional Bi-Directional LSTM (C-BiLSTM)](#convolutional-bi-directional-lstm-c-bilstm)
  - [Results](#results)
  - [Conclusions](#conclusions)
  - [References](#references)
  - [Appendix](#appendix)
    - [A. Spelling replacements](#a-spelling-replacements)

## Abstract

## Introduction

The goal of this project is to attempt to solve a toxicity classification problem in the field of natural language processing (NLP). The approach used in this paper discusses neural networks and how they perform for this classification task. We discuss our reasoning, expectations and thoughts of the execution of our project. The toxicity classification problem at hand is due to the increasing amount of discussion platforms such as, comment feeds on live broadcasts and other fora. Some live broadcasts are targeted for wide audiences and need efficient, near instantanious, profanity filtering.

Typically, the issue with naïve profanity filtering is that it focuses too heavily on individual words while ignoring the context. It was studied that names of the frequently attacked identities were automatically accociated to toxicity by machine learning (ML) models, even if the individual(s) themselves, or the context, were not offensive. (Kaggle. 2019)

### Scope of the project

This paper is for a Kaggle competition ([Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/ "Competition page")). The competition provides a grader that evaluates the quality of the resulting model. The overall assessment of the goodness of our model is deviced with an overall Area under the curve (AUC) Receiver operating characteristic curve (ROC) test and with multiple other submetrics. The submetrics are:

- Bias AUCs: To prevent unintended bias we use three specific subsets of the test for each identity, attemping to capture all the aspects of unintended bias.
  - Subgroup AUC: Focuses on specific identity subgroup
  - Background Positive, Subgroup negative (BSPSN) AUC: Evaluates only non-toxic examples for an identity and toxic examples without the identity
  - Background Negative, Subgroup positive (BNSP) AUC: The test set is restricted opposite to BSPN, only featuring examples of toxic examples of an identity and non-toxic examples without the identity
- Generalized Mean of Bias AUCs
  - Calculates the combination of the per-identity Bias AUCs into one overall measure with:

$$M_p(m_s) = \left(\frac{1}{N} \sum_{s=1}^{N} m_s^p\right)^\frac{1}{p}$$
where:

- $M_p$ = the pth power-mean function
- $m_s$ = the bias metric m calulated for subgroup s
- $N$ = number of identity subgroups

Lastly, after obtaining the overall AUC and the General Mean of Bias AUCs the calculation of the final model is done with formula:
$$score = w_0 AUC_{overall} + \sum_{a=1}^{A} w_a M_p(m_{s,a})$$
where:

- $A$ = number of submetrics (3)
- $m_s,a$ = bias metric for identity subgroup $s$ using submetric $a$
- $w_a$ = a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

## Imports and Utilities

```{python}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import re # regexes
import matplotlib.pyplot as plt # plotting
import os
import time

from keras import preprocessing
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import os

# custom helper libraries (provided with the project)
from utils import get_individual_words

device = torch.device("cpu")

np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))
torch.backends.cudnn.deterministic = True
```

## Analysing the datasets

The origin of the data is from
year 2017 when the Civil Comments platform shut down and published their ~2m public comments making them available for researchers. Each item in the dataset  has one primary attribute for the toxicity, `target`, indicating a goal value that the models should try to achieve. The trained model should then predict the `target` toxicity for the test data.

In addition to `target`, there are several subtypes of toxicity. These are not supposed to be predicted by the model, but they are for providing additional avenue for future research. The subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Along with these, it is notable that the attribute `parent_id` could be used for training a model. The reason for this is that we think that the neural network should mark some difference between comments that start a thread versus ones that do not.

Some of the comments have a label for identity. There are multiple identity attributes, each representing the identity that *is mentioned* in the comment. The identities we are interested, as the ones used in the validation of the model, are:

- male
- female
- homosexuality_gay_or_lesbian
- christian
- jewish
- muslim
- black
- white
- psychiatric_or_mental_illness

Based on this short task description analysis, we made the decision to select the following columns for training the model and drop all other columns:

```{python}
identity_columns = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness"
]

relevant_columns = [
    "id",
    "target",
    "comment_text"
] + identity_columns
```

### Cleaning up the data

The data is a little noisy. Example given, there are some reoccurring special characters and the dataset contains duplicate comments with same content. However, the different comments may have been labelled with different targets or subgroups [kaggle].

The operations we do for the datasets are; lowercase the words, remove the non-alpha characters, fill empty values with 0 and preprocess the data so that we obtain a much smaller, more compact, sets for the training and validation.

```{python}
def clean_sentence(sentence):
    return re.sub(
        "[^a-z\d\s]", # Remove all non-alphanumeric or space characters from comment text
        " ",
        " ".join(map(
            lambda word: replace_spelling[word] if word in replace_spelling else word,
            sentence.lower().split()
        )).replace("'", "") # ... and drop all remaining ' -marks (as we do not want to replace them with spaces)
    )

def create_cleaned_file(from_name, to_name, cols, drop_duplicates):
    """Create a cleaned file from a file"""
    data = pd.read_csv(
        "../input/jigsaw-unintended-bias-in-toxicity-classification/{}.csv".format(from_name),
        usecols=cols,# use only relevant columns, as specified before in the notebook
    )
    data.set_index("id", inplace=True)

    _, original_word_count = get_individual_words(data["comment_text"])

    data["comment_text"] = data["comment_text"].transform(clean_sentence)
    data = data.fillna(0) # fill empty values with 0

    if drop_duplicates:
        cleaned_words = set()
        data["comment_text"].str.split().apply(cleaned_words.update)

        # Write summary file of the cleanup process
        pd.DataFrame({
            "previous_word_count": [original_word_count],
            "cleaned_word_count": [get_individual_words(data["comment_text"])[1]],
            "previous_row_count": [len(data)],
            "cleaned_row_count": [data['comment_text'].nunique()]
        }).to_csv("./"+to_name+"_summary.csv")

        data = data.groupby('comment_text').mean().reset_index()
    data.to_csv("./"+to_name+".csv", index_label="id")

def print_cleanup_summary(filename):
    # read cleanup summary from saved file
    cleanup_summary = pd.read_csv("./"+filename+"_summary.csv")
    initial_row_count, initial_word_count = cleanup_summary.loc[
        0,
        ["previous_row_count", "previous_word_count"]
    ]
    cleaned_row_count, cleaned_word_count = cleanup_summary.loc[
        cleanup_summary.index[-1],
        ["cleaned_row_count", "cleaned_word_count"]
    ]

    print("The original data was reducted by {} rows ({:.2f}%) and by {} words ({:.2f}%)".format(
        initial_row_count - cleaned_row_count,
        100 * (1 - cleaned_row_count / initial_row_count),
        initial_word_count - cleaned_word_count,
        100 * (1 - cleaned_word_count / initial_word_count)
    ))

def read_cleaned_file(from_name, to_name, cols = None, drop_duplicates = True):
    if True or not os.path.isfile("./{}.csv".format(to_name)):
        create_cleaned_file(from_name, to_name, cols, drop_duplicates)
    # read data from cleaned data file if not already set
    return  pd.read_csv("./{}.csv".format(to_name))
```

```{python}
print("reading train data...")
train_data = read_cleaned_file(
    'train',
    'train_cleaned',
    relevant_columns # discards all other columns
)
print_cleanup_summary("train_cleaned")


print("reading test data...")
test_data = read_cleaned_file(
    'test',
    'test_cleaned',
    drop_duplicates = False # we don't want to drop duplicates, as every comment in test_data have to be labelled
)
print("The test data is read to memory")
```

### Visualizations of the data

```{python}
# Import visualization functions
from visualizations import (
    visualize_toxicity_by_identity,
    visualize_target_distribution,
    visualize_comment_length
)
```

In this section we will attempt to find further details of the data with some basic exploratory data analysis (EDA) techniques.

Now that we have a cleaned version of the datasets we can further analyze and visualize them. Lets first look at the binary distribution of toxic and non-toxic inputs in the dataset. A message is toxic whenever its toxicity (attribute `target`) is larger than the data toxicity threshold (`.5`).

```{python}
def calculate_binary_distribution(data):
    """Calculate percentage of toxic and non-toxic entries"""
    toxic = 0
    non_toxic = 0
    for value in data["target"]:
        if value >= .5:
            toxic += 1
        else:
            non_toxic += 1
    data_count = len(data)
    return (toxic / data_count, non_toxic / data_count)
```

```{python}
count_toxic, count_non_toxic = calculate_binary_distribution(train_data)

# Print binary toxicity distribution
print("Amount of toxic comments {:.2f}% and non-toxic comments {:.2f}%.".format(
    count_toxic,
    count_non_toxic
))
```

Next a generic visualization showing the distribution of toxic and non-toxic messages in the dataset.

```{python}
visualize_target_distribution(train_data)
```

Then toxicity distributions by identity groups.

```{python}
visualize_toxicity_by_identity(train_data, identity_columns)
```

```{python}
train_data.sort_values(by=["target", "jewish"], ascending=False).head(10)
```

Here, we see the comment lengths (number of characters in a sentence) for the train and validation datasets. The distribution is bimodal but heavily skewed to shorter lengths. The two peaks are at approximately 100 and 1000 character marks. There are only a total of four (4) items in the training dataset with more-than or equal to 1250 characters in the datasets. The corresponding number for the test dataset is three (3).

```{python}
def count_more_than_equal(data, attr, threshold):
    try:
        threshold = float(threshold)
        count = 0
        for value in data[attr]:
            if isinstance(value, str):
                value = len(value)
            if value >= threshold:
                count += 1
        return count
    except ValueError:
        return 0
```

```{python}
threshold = 1250
print("Amount of {} or more character sentences.\ntrain data: {}\ntest data: {}".format(
    threshold,
    count_more_than_equal(train_data, "comment_text", threshold),
    count_more_than_equal(test_data, "comment_text", threshold)
))
```

```{python}
visualize_comment_length(train_data, "Comment text lengths for train data")
visualize_comment_length(test_data, "Comment text lengths for test data")
```

As we can see from the illustrations above, the two datasets have very similar shape and the model should hence yield good results for both the training and the test dataset.


## Methods

Natural language processing (NLP) is an example of a Supervised Machine Learning task that focuses in labelled datasets containing sequences and or single words. The purpose of NLP is to train a classifier that can distinguish the sets of words into their right belonging categories (classes).

Typically the text classification pipeline contains the following components:

- Training data: Input text for the model
- Feature vector: A vector that describe the input texts, in some charasteristic or multiple ones
- Labels: The classes that we plan to predict with our trained model
- Algorithm: A machine learning algorithm that is used to classify the inputs
- Model: The result of the training, this will perform the label predictions (https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f)

A prime example of a text classification problem is detection of spam or toxic sentences.

Harish Yenala, Ashish Jhanwar, Manoj K. Chinnakotla and Jay Goyal suggest in their paper *Deep learning for detecting inappropriate content in text* to utilize a combination of Convolutional Neural Networks (CNN) and Bi-directional Long short-term memory networks (C-BiLSTM) to achieve a high performing classifier for a task very similar to ours.

More generally, use of recursive neural networks (RNN), LSTM being an example of these, in natural language processing (NLP) is an intuitive and an advisable method. RNNs exceed in grammar learning. This is mainly because their ability to process sequences of inputs, such as, in the case of toxicity filtering, sequences of words.

### Preprocess the data

In the preprocessing we want to perform integer encoding, so that each word is represented by a unique integer. Doing so allows us to later on implement an embedding layer that we need for processing text data.

The purpose of the word embedding is to obtain a dense vector representation of each word. This vector is then capable of, example given, capturing the context of that word, the semantic and syntactic similarity and relation to other words [A Word Embedding based Generalized Language Model for Informal Retrieval pp.795].

```{python}
train_x = train_data["comment_text"]
train_y = train_data[["target"] + identity_columns]
```

```{python}
sentences = train_x.append(test_data["comment_text"])

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
```

### Convolutional Neural Network (CNN)

CNNs are feed-forward artificial neural networks. They use a variation of multilayer perceptrons that offer minimal preprocessing. Use of CNNs in NLP is a relatively new technique as previously their primary use case was in computer vision. Use of CNN models for NLP tasks have shown to be effective in semantic parsing **[Yih et al., 2014. Semantic Parsing for Single-Relation Question Answerings of ACL 2014]**, search query retrieval **[Shen et al., 2014. Learning Semantic Representations Using Convolutional Neural Networks for Web Search. In Proceedings of WWW 2014.]**, sentence modeling **[Kalchbrenner et al., 2014. A Convolutional Neural Network for Modelling Sentences, In Proceedings of ACL 2014.]**, and other, more traditional NLP tasks **[Collobert et al., 2011. Natural Language Processing (Almost) from Scratch. Journal of Machine Learning Research 12:2493-2537]**.

Harrison Jansma recommends in his Medium article not to use Dropout **[Harrison J. 2018. Don't Use Dropout in Convolutional Networks. Medium.]**. Dropout is a technique for preventing overfitting in neural networks by randomly drop units (and their connections) from neural network during training **[Srivastana et al., 2014. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research 15:1929-1958]**. According to Jansma, there are two main reasons why use of dropout is declining in CNNs. First, use of dropout is less effective in regularizing convolutional layers in contrast to batch normalization. Secondly, dropout is good for fully connected layers but more recent architectures have moved away from these fully-connected blocks. Hence, it is not the tool for these architectures.


### Long short-term memory (LSTM)

LSTM is a recurrent network architecture in conjunction with an efficient, gradient-based learning algorithm. All together LSTM is able to store information over extended time intervals while resolving problems associated with backpropagation through time (BPTT) and real-time recurrent learning (RTRL). **(Hochreiter and Schmidhuber, 1995)[Hochreiter, S. and Schmidhuber, J. (1995). Long short term memory. München: Inst. für Informatik.]**

The greatest advantage of using LSTM for NLP is its ability to handle noise, distributed representations, and continuous values. LSTMs ability to function without a finite number of states enables it to hence work for any sequence lengths and items in the sequence, as long as these are preprocessed to numbers. **(Hochreiter and Schmidhuber, 1995)**

#### The network

Our LSTM network consists of embedding and linear layers along with the LSTM defined by PyTorch. The PyTorch LSTM applies a multi-layer long short-term memory RNN to the input sequence. The computation, that the layer does, for each input sequence is:

$i_t = \sigma(W_{ii}x_t + b_{ii} + W_{hi}h_{(t-1)} + b_{hi})$

$f_t = \sigma(W_{if}x_t + b_{if} + W_{hf}h_{(t-1)} + b_{hf})$

$g_t = \tanh(W_{ig}x_t + b_{ig} + W_{hg}h_{(t-1)} + b_{hg})$

$o_t = \sigma(W_{io}x_t + b_{io} + W_{ho}h_{(t-1)} + b_{ho})$

$c_t = f_t * c_{(t-1)} + i_t * g_t$

$h_t = o_t * \tanh(c_t)$

where $h_t$ is the hidden state at time $t$, $c_t$ is the cell state at time $t$, $x_t$ is the input time at $t$, $h_{(t-1)}$ is the hidden state of the layer at time $t-1$ or the initial hidden state at time $o$, and $i_t$, $f_t$, $g_t$ and $o_t$ are the input, forget, cell, and output gates, respectively. $\sigma$ is the sigmoind function, and $*$ is the Hadamard product. **[PyTorch Documentation]**

```{python}

class LSTMNetwork(nn.Module):
    def __init__(self, max_len, vocab_size, hidden_size, out_size):
        super(LSTMNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, max_len)

        self.pre = nn.Sequential(
            nn.Linear(max_len, hidden_size * 2),
            nn.Dropout(.20),
            nn.ELU(),
            nn.Linear(hidden_size * 2, max_len)
        )

        self.lstm = nn.LSTM(
            input_size = max_len,
            hidden_size = hidden_size,
            bidirectional = True,
            batch_first = True,
            num_layers = 4
        )

        self.net = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ELU(),
            nn.Linear(hidden_size * 2, out_size)
        )

    def forward(self, x):
        embedding = self.embedding(x)
        out = self.pre(embedding)
        out, _ = self.lstm(embedding)

        out = torch.cat((
            torch.max(out, 1)[0], # global max pooling
            torch.mean(out, 1) # global average pooling
        ), 1)

        return self.net(out)
```

```{python}
max_len = len(sentences.max())

lstm = LSTMNetwork(max_len, len(tokenizer.word_index) + 1, 128, len(train_y.columns)).to(device)
criterion = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = optim.Adam(lstm.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .7 ** epoch)
```

```{python}
log_nth = 200
epochs = 4

print("training model...")

train_tokens = preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(train_x),
    maxlen = max_len,
)

train_data_tensor = torch.tensor(train_tokens, dtype=torch.long).to(device)
train_labels_tensor = torch.tensor(train_y.values, dtype=torch.float32).to(device)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor),
    batch_size=512,
    shuffle=True
)

for epoch in range(epochs):
    epoch_start = time.time()
    scheduler.step()
    lstm.train() # set lstm to train mode
    total_loss = 0
    for i, (x, labels) in enumerate(train_loader, 1):
        x.to(device)

        optimizer.zero_grad()

        out = lstm(x)

        loss = criterion(out, labels)
        total_loss += loss.item()
        loss.backward()

        optimizer.step()

        if i % log_nth == 0:
            print(total_loss / i, "{} / {} (batches ran / left to run)".format(i, len(train_loader)))

    print(
        "epoch: {}/{}".format(epoch + 1, epochs),
        "loss: {:.4f}".format(total_loss / len(train_loader)),
        "time passed: {}".format(time.time() - epoch_start)
    )

# Save trained model state in case something goes south
print("saving model...")
torch.save(lstm.state_dict(), "./lstm.pd")
```

```{python}
lstm.eval() # set lstm to evaluation mode

test_tokens = preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(test_data["comment_text"]),
    maxlen = max_len,
)

chunk_size = 1000
chunks = len(test_tokens) // chunk_size + 1

predictions = pd.DataFrame(columns=["id", "comment", "prediction"])

for i in range(chunks):
    print("{}/{}".format(i + 1, chunks))
    first, last = i * chunk_size, min(len(test_tokens), (i + 1) * chunk_size)

    test_data_tensor = torch.tensor(test_tokens, dtype=torch.long)[first:last].to(device)

    predictions = pd.concat([
        predictions,
        pd.DataFrame({
            'id': test_data['id'][first:last],
            'prediction': torch.sigmoid(
                lstm(test_data_tensor).detach().cpu() # need to copy the memory to cpu before casting to numpy array
            )[:, 0]
        })
    ], sort=False)

assert len(predictions) == len(test_tokens)
predictions[["id", "prediction"]] \
    .to_csv('./submission.csv', index=False)
```

### Convolutional Bi-Directional LSTM (C-BiLSTM)

A combination network of Convolutional Neural Network and Bi-Directional LSTM.

```{python}
class LSTMNetwork(nn.Module):
    def __init__(self, max_len, vocab_size, hidden_size, out_size):
        super(LSTMNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, max_len)

        self.conv = nn.Sequential( # [1024, 175, 175])
            nn.Conv1d(max_len, hidden_size, 3, padding = 1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding = 1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding = 1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size = max_len,
            hidden_size = hidden_size,
            bidirectional = True,
            batch_first = True,
            num_layers = 2
        )

        self.net = nn.Sequential(
            nn.Linear(hidden_size * hidden_size * 2, hidden_size * 2),
            nn.ELU(),
            nn.Linear(hidden_size * 2, out_size)
        )

    def forward(self, x):
        embedding = self.embedding(x)
        out, _ = self.lstm(self.conv(embedding))

        out = torch.flatten(out, start_dim=1)

        return self.net(out)
```

## Results

We tried above networks with tweaking the parameters and network structures. However, due to limited time reserved to project and computational resources we were not able to try enough alterations.

| activation fn | score   | lr coefficient | dropout | hidden size | lstm_layers |
|---------------|---------|----------------|---------|-------------|-------------|
| ELU           | 0.91227 | .7             | .25     | 128         | 2           |
| ReLU          | 0.89617 | .5             | .25     | 128         | 2           |
| ReLU          | 0.90599 | .7             | .25     | 128         | 4           |

## Conclusions

## References

Yenala, H., Jhanwar, A., Chinnakotla, M.K. et al. Int J Data Sci Anal (2018) 6: 273. [](https://doi.org/10.1007/s41060-017-0088-4)

https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

## Appendix

### A. Spelling replacements

```{python}
replace_spelling = {
    "I'd": 'I would',
    "I'll": 'I will',
    "I'm": 'I am',
    "I've": 'I have',
    "ain't": 'is not',
    "aren't": 'are not',
    "can't": 'cannot',
    'colour': 'color',
    "could've": 'could have',
    "couldn't": 'could not',
    "didn't": 'did not',
    "doesn't": 'does not',
    "don't": 'do not',
    "hadn't": 'had not',
    "hasn't": 'has not',
    "haven't": 'have not',
    "he'd": 'he would',
    "he'll": 'he will',
    "he's": 'he is',
    "here's": 'here is',
    "how's": 'how is',
    "i'd": 'i would',
    "i'll": 'i will',
    "i'm": 'i am',
    "i've": 'i have',
    "isn't": 'is not',
    "it'll": 'it will',
    "it's": 'it is',
    "let's": 'let us',
    "might've": 'might have',
    "must've": 'must have',
    "she'd": 'she would',
    "she'll": 'she will',
    "she's": 'she is',
    "shouldn't": 'should not',
    "that's": 'that is',
    'theatre': 'theater',
    "there's": 'there is',
    "they'd": 'they would',
    "they'll": 'they will',
    "they're": 'they are',
    "they've": 'they have',
    'travelling': 'traveling',
    "wasn't": 'was not',
    "we'd": 'we would',
    "we'll": 'we will',
    "we're": 'we are',
    "we've": 'we have',
    "weren't": 'were not',
    "what's": 'what is',
    "where's": 'where is',
    "who'll": 'who will',
    "who's": 'who is',
    "who've": 'who have',
    "won't": 'will not',
    "would've": 'would have',
    "wouldn't": 'would not',
    "you'd": 'you would',
    "you'll": 'you will',
    "you're": 'you are',
    "you've": 'you have'
}
```