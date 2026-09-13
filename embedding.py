import torch
import torch.nn as nn
class Embedding(nn.Module):
    def __init__(self,en_vocabulary:dict):
        super().__init__()
        self.emb_dimension = 64
        self.vocab_size = len(en_vocabulary)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.emb_dimension)
        self.linear_model = nn.Sequential(
            nn.Linear(self.emb_dimension,8*self.emb_dimension),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8*self.emb_dimension,8*self.emb_dimension),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8*self.emb_dimension,self.emb_dimension),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.emb_dimension,self.vocab_size)
            )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.01)

    def forward(self,x):
        output = self.embedding(x)
        output = self.linear_model(output)
        return output