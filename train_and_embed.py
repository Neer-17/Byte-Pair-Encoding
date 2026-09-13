#Imports
import torch
import numpy as np
from tokenizer import Tokenizer
from embedding import Embedding
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Read the text file
with open(r'example.txt','r+',encoding='utf8') as file:
    text = file.read()

#Create tokens
tokenizer = Tokenizer(text=text,merges=1000,min_pair_freq=2)
vocab,merges = tokenizer.tokenize()
en_text = tokenizer.encode(text)
en_vocab = tokenizer.encoded_vocab
de_vocab = tokenizer.decoded_vocab

#Make pair function
def make_pairs(en_text,context_size):
    target_context_pairs = []
    for i in range(context_size,len(en_text)-context_size):
        target = en_text[i]
        context_words = en_text[i-context_size:i]
        context_words.extend(en_text[i+1:i+1+context_size])
        for word in context_words:
            target_context_pairs.append((target,word))

    return target_context_pairs

#Get batch function
def get_batch(X_train,Y_train,batch_size):
    st_index = np.random.randint(0,len(X_train)-batch_size)
    x = X_train[st_index:st_index+batch_size]
    y = Y_train[st_index:st_index+batch_size]
    return x,y

#Visualize Word Embedding Function
def visualize_words_embedding(model,en_words,words,n_cluster):


  # extract the model weights
  word_vectors = model.embedding.weight.data

  # get the word embeddings
  word_vectors = word_vectors[en_words]
  word_vectors = word_vectors.cpu().numpy()

  # fit a 3d t-SNE model to the embedding vectors
  tsne = TSNE(n_components=3,perplexity=20,random_state=42)
  word_vectors_3d = tsne.fit_transform(word_vectors)

  # get the clusters
  kmeans = KMeans(n_clusters=n_cluster,random_state=42,n_init="auto")
  kmeans.fit(word_vectors)
  cluster_labels = kmeans.predict(word_vectors)

  # show the figure
  fig = px.scatter_3d(x=word_vectors_3d[:,0],y=word_vectors_3d[:,1],z=word_vectors_3d[:,2],color=cluster_labels,title="Word Embeddings Visualization",text=words)
  fig.show()

  # Save the figure
  fig.write_html(f'word_embeddings.html')

#Create vectors
pairings = make_pairs(en_text,context_size=5)

X_train = []
Y_train = []
X_val = []
Y_val = []
for target, context in pairings[:int(0.9 * len(pairings))]:
    X_train.append(target)
    Y_train.append(context)

for target, context in pairings[int(0.9 * len(pairings)):]:
    X_val.append(target)
    Y_val.append(context)

X_train = torch.tensor(X_train,dtype=torch.long)
Y_train = torch.tensor(Y_train,dtype=torch.long)
X_val =  torch.tensor(X_val,dtype=torch.long)
Y_val = torch.tensor(Y_val,dtype=torch.long)

X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_val = X_val.to(device)
Y_val = Y_val.to(device)

#Train function
def train(model,X_train,Y_train,X_val,Y_val,batch_size):
        num_batches = len(X_train)//batch_size
        for epoch in range(model.epochs):
            total_loss = 0
            for i in range(0,len(X_train),batch_size):
                # print(f"Epoch: {epoch} | Batch: {i//batch_size}")
                x,y = get_batch(X_train,Y_train,batch_size)
                x = x.to(device)
                y = y.to(device)
                model.optimizer.zero_grad()
                y_pred = model.forward(x)
                y_true = y.view(-1)
                loss = model.criterion(y_pred,y_true)
                total_loss += loss
                loss.backward()
                model.optimizer.step()
            if (epoch%10 == 0):
                model.eval()

                with torch.no_grad():
                    x,y = get_batch(X_val,Y_val,batch_size)
                    x = x.to(device)
                    y = y.to(device)
                    output = model.forward(x)
                    y = y.view(-1)
                    val_loss = model.criterion(output,y)
                    print(f"Epoch: {epoch} | Train loss: {total_loss/num_batches:.4f} | Val loss: {val_loss:.4f}")
                model.train()

#Parameters
def train(model,X_train,Y_train,X_val,Y_val,batch_size,epochs):
        num_batches = len(X_train)//batch_size
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0,len(X_train),batch_size):
                # print(f"Epoch: {epoch} | Batch: {i//batch_size}")
                x,y = get_batch(X_train,Y_train,batch_size)
                x = x.to(device)
                y = y.to(device)
                model.optimizer.zero_grad()
                y_pred = model.forward(x)
                y_true = y.view(-1)
                loss = model.criterion(y_pred,y_true)
                total_loss += loss
                loss.backward()
                model.optimizer.step()
            if (epoch%10 == 0):
                model.eval()

                with torch.no_grad():
                    x,y = get_batch(X_val,Y_val,batch_size)
                    x = x.to(device)
                    y = y.to(device)
                    output = model.forward(x)
                    y = y.view(-1)
                    val_loss = model.criterion(output,y)
                    print(f"Epoch: {epoch} | Train loss: {total_loss/num_batches:.4f} | Val loss: {val_loss:.4f}")
                model.train()

#Parameters
epochs = 51
batch_size = 512

#Embedding model
model = Embedding(en_vocab).to(device)

#Train
train(model,X_train,Y_train,X_val,Y_val,batch_size,epochs)

# get encoded_words and decoded_words
en_words = list(en_vocab.values())
words = []
for word in en_words:
    words.append(tokenizer.decode([word]))

# Visualize Word Embedding
visualize_words_embedding(model,en_words,words,n_cluster=5)