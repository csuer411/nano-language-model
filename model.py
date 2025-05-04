import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer

def precompute_cls(max_len, dim):
    """precompute the position encoding
    Args:
        max_len: 最大序列长度
        dim: 嵌入维度
    Returns:
        pe: 位置编码张量，形状为 [max_len, dim]
    """
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def transpose_mha(x, n_head):
    """apply transpose operation for q,k,v"""
    b, l, d = x.shape
    x = x.view(b, l, n_head, -1).transpose(1,2)
    return x

class FeedFoward(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 2*hidden_dim)
        self.up = nn.Linear(hidden_dim, 2*hidden_dim)
        self.down = nn.Linear(2*hidden_dim, hidden_dim)
        self.act = nn.SiLU()
    
    def forward(self, x):
        proj = self.proj(x)
        up = self.up(x)
        gate = self.act(proj*up)
        out = self.down(gate)
        return out
        
class MultiheadAttention(nn.Module):

    def __init__(self, n_head, input_dim, max_len):
        super().__init__()
        assert input_dim % n_head == 0
        self.n_head = n_head
        self.wq = nn.Linear(input_dim, input_dim)
        self.wk = nn.Linear(input_dim, input_dim)
        self.wv = nn.Linear(input_dim, input_dim)
        self.wo = nn.Linear(input_dim, input_dim)
        mask = torch.ones((max_len, max_len)).triu(1)
        self.register_buffer('mask', mask)

    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q, k, v = transpose_mha(q, self.n_head), transpose_mha(k, 
                            self.n_head), transpose_mha(v, self.n_head)
        #b,n_head,l,dim//n_head
        _, _, l, divide_dim = q.shape
        attn_map = q @ k.transpose(2, 3) / math.sqrt(divide_dim)
        attn_map = torch.softmax(attn_map.masked_fill(self.mask[:l, :l]==1, float("-inf")),dim=-1)
        out = attn_map @ v
        out = out.transpose(1,2).contiguous().view(x.size(0), x.size(1), -1)
        out = self.wo(out)
        return out

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 计算均方根并应用缩放
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(norm + self.eps)
        out = x * rsqrt * self.gamma
        return out

class Decoder(nn.Module):

    def __init__(self, n_layer, n_dim, n_head, n_vocabulary, max_len):
        super().__init__()
        self.n_layer = n_layer
        self.words = nn.Embedding(n_vocabulary, n_dim)
        self.position_encoding = precompute_cls(max_len, n_dim).to("cuda")
        self.logits = nn.Linear(n_dim, n_vocabulary)
        self.mhas = nn.ModuleList([MultiheadAttention(n_head, n_dim, max_len) for _ in range(n_layer)])
        self.ffs = nn.ModuleList([FeedFoward(n_dim) for _ in range(n_layer)])
        self.norms = nn.ModuleList([RMSNorm(n_dim) for _ in range(2*n_layer)])

    def forward(self, input_ids):
        b, l = input_ids.shape
        embeddings = self.words(input_ids)
        # 添加位置编码
        pos_enc = self.position_encoding[:l, :]
        embeddings = embeddings + pos_enc
        for i in range(self.n_layer):
            embeddings = embeddings + self.mhas[i](self.norms[2*i](embeddings))
            embeddings = embeddings + self.ffs[i](self.norms[2*i+1](embeddings))
        out = self.logits(embeddings)
        return out

class NLM(nn.Module):

    def __init__(self, n_layer, n_dim, n_head, n_vocabulary, max_len):
        super().__init__()
        self.decoder = Decoder(n_layer, n_dim, n_head, n_vocabulary, max_len)
        self.tokenizer = AutoTokenizer.from_pretrained("tokenizers")

    def forward(self, input_ids):
        out = self.decoder(input_ids)
        return out

    def compute_loss(self, input_ids, target_ids):
        out = self.forward(input_ids)
        loss = nn.CrossEntropyLoss(ignore_index=0)(out.reshape(-1, 6400), target_ids.reshape(-1))
        return loss
        
    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
        
    @torch.no_grad()
    def generate(self, input_ids, temperature=1, max_len=512, end_token=2):
        while input_ids[0][-1] != end_token and len(input_ids[0])<= max_len:
            out = self.forward(input_ids)
            # 获取最后一个token的logits
            logits = out[:,[-1],:]
            
            if temperature == 0.0:
                # 当温度为0时，直接使用argmax（确定性选择）
                next_token = torch.argmax(logits, dim=-1)
            else:
                # 应用温度缩放
                scaled_logits = logits / temperature
                # 计算概率分布
                probs = torch.softmax(scaled_logits, dim=-1)
                # 根据概率分布采样下一个token
                next_token = torch.multinomial(probs[0,0], 1).unsqueeze(0)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return self.tokenizer.decode(input_ids[0])
        
        
        
