# personal implementation of a lecture given by Andrej Karpathy
# find his GitHub here - https://github.com/karpathy
# Tutorial lecture here - https://youtu.be/kCc8FmEb1nY?si=nqz7CfUcjVK3XvBC
# This is training on the complete poetic works of Edgar Allan Poe instead of Shakespear in his
# tutorial

# library we will use for computation, also need to pip install numpy
import torch
import torch.nn as nn
from torch.nn import functional as F

print(f"Torch Version: {torch.__version__}, GPU: {torch.cuda.is_available()}, NUM_GPU: {torch.cuda.device_count()}")

block_size = 256  # how many independent sequences will we process in parallel?
batch_size = 64  # what is the maximum content length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 200
torch.manual_seed(1337)
force_cpu = False
n_embed = 384
n_heads = 6
n_layers = 6
dropout = 0.2
train_model = False


# set the device to use GPUs if available (mps = GPU for Apple Silicon)
if torch.backends.mps.is_available() and not force_cpu:
    device = torch.device('mps')
    print("Apple GPU is available and PyTorch is configured to use MPS")
elif torch.cuda.is_available() and not force_cpu:
    device = torch.device('cuda')
    print("CUDA is available and PyTorch is configured to use CUDA")
else:
    print("GPU is not available and PyTorch is configured to use CPU")
    device = torch.device('cpu')

# We import our training data here
with open('./shakespear.txt', 'r') as f:
    text = f.read()

# get the unique characters from the training data, and the size of the set
chars = sorted(list(set(text)))
vocab_size = len(chars)

# simple encoding from char to int and back for our vocabulary
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(x): return [stoi[c] for c in x]


def decode(x): return ''.join([itos[i] for i in x])


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    #  multiple heads of self-attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_heads) for _ in range(n_layers)])
        self.ln1 = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensors of integers
        token_embedded = self.token_embedding_table(idx)  # (B,T,C)
        position_embedded = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = token_embedded + position_embedded  # (B,T,C)
        x = self.blocks(x)
        local_logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            local_loss = None
        else:
            B, T, C = local_logits.shape
            local_logits = local_logits.view(B * T, C)
            targets = targets.view(B * T)
            local_loss = F.cross_entropy(local_logits, targets)
        return local_logits, local_loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            local_logits, local_loss = self(idx_cond)
            # focus on only the last time step
            local_logits = local_logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(local_logits, dim=-1)  # (B ,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# put our training data into pytorch
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

# Split into training and verification data sets
# we'll use 90% for training
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    local_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(local_data) - block_size, (batch_size,))
    x = torch.stack([local_data[i:i + block_size] for i in ix])
    y = torch.stack([local_data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        local_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            local_logits, local_loss = model(X, Y)
            local_losses[k] = local_loss.item()
        out[split] = local_losses.mean()
    model.train()
    return out



if train_model:
    model = BigramLanguageModel()
    m = model.to(device)
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iteration in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iteration % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # save the trained model so it can be reused
    torch.save(model.state_dict(), "shakespear_model.pt")
else:
    print('loading model')
    model = BigramLanguageModel()
    model.load_state_dict(torch.load("shakespear_model.pt", weights_only=True))
    model.eval()
    m = model.to(device)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))
