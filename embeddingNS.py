import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, emb_dimension=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension

        
        self.target_embedding = nn.Embedding(vocab_size, emb_dimension)
        self.context_embedding = nn.Embedding(vocab_size, emb_dimension)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

    def forward(self, target_ids, pos_context_ids, neg_context_ids):
        # target_ids: (batch,)
        # pos_context_ids: (batch,)
        # neg_context_ids: (batch, k)

        t = self.target_embedding(target_ids)              # (batch, dim)
        c_pos = self.context_embedding(pos_context_ids)     # (batch, dim)
        c_neg = self.context_embedding(neg_context_ids)     # (batch, k, dim)

        pos_score = torch.sum(t * c_pos, dim=1)             # (batch,)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(c_neg, t.unsqueeze(2)).squeeze(2)  # (batch, k)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)            # (batch,)

        loss = -(pos_loss + neg_loss).mean()
        return loss