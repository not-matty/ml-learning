import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class Trigram(nn.Module):

    def __init__(self):
        super().__init__()
        self.words = open('part1/names.txt', 'r').read().splitlines()

        self.chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s:i+1 for i,s in enumerate(self.chars)} # characters to indices
        self.stoi['.'] = 0
        self.size = len(self.stoi)
        self.W = torch.randn((10, 27), requires_grad=True) # 2 * embedding size (concat of two embeddings)
        self.embedding = nn.Embedding(self.size, 5) # shape: num chars, ~sqrt(num chars)
    def create_train(self):

        xs, ys = [], []
        for w in self.words[:]:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                ix3 = self.stoi[ch3]
                print(ch1, ch2)
                xs.append((ix1, ix2))
                ys.append(ix3)

        xs = torch.tensor(xs, dtype=torch.long)
        ys = torch.tensor(ys, dtype=torch.long)
        return xs, ys

    def forward(self, xs):

        emb1 = self.embedding(xs[:, 0])  # first character embedding
        emb2 = self.embedding(xs[:, 1])  # second

        # concatenate
        xenc = torch.cat([emb1, emb2], dim=1)
        logits = xenc @ self.W
        return logits

    def cross_entropy_loss(self, logits, ys):
        # subtract max for numerical stability
        logits = logits - logits.max(dim=1, keepdim=True).values
        # compute log-softmax
        log_probs = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True))
        target_log_probs = log_probs.gather(dim=1, index=ys.unsqueeze(1))
        loss = -target_log_probs.mean()
        return loss + 0.01 * (self.W**2).mean()
    
    def train(self, iterations, xs_train, ys_train, xs_test, ys_test, lr):
        loss_history = []
        test_loss_history = []
        k_history = []
        for k in range(iterations):

            logits = model.forward(xs_train)
            loss = model.cross_entropy_loss(logits, ys_train)
            if self.W.grad is not None:
                self.W.grad = None
            loss.backward()
            model.W.data += -lr * model.W.grad
            
            if k % 10 == 0:
                loss_history.append(loss.item())
                
                with torch.no_grad():  # avoid grad on test set
                    test_logits = self.forward(xs_test)
                    test_loss = self.cross_entropy_loss(test_logits, ys_test)
                    test_loss_history.append(test_loss.item())

                k_history.append(k)
                print('iteration:', k, 'loss:', loss.item())

        return loss_history, test_loss_history, k_history

    def accuracy(self, xs, ys):

        logits = self.forward(xs)
        #argmax or multinomial?
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == ys).sum().item()
        accuracy = correct / ys.size(0) * 100
        return accuracy
    
    def train_test_split(self, xs, ys, test_size=0.2, random_seed=42):
        torch.manual_seed(random_seed)
        indices = torch.randperm(xs.size(0))
        test_split = int(xs.size(0) * test_size)
        test_indices = indices[:test_split]
        train_indices = indices[test_split:]
        xs_train, ys_train = xs[train_indices], ys[train_indices]
        xs_test, ys_test = xs[test_indices], ys[test_indices]
        return xs_train, ys_train, xs_test, ys_test
    
model = Trigram()
xs, ys = model.create_train()
xs_train, ys_train, xs_test, ys_test = model.train_test_split(xs, ys)

iterations = 1000
lr = 0.5
loss_history, test_loss_history, k_history = model.train(iterations, xs_train, ys_train, xs_test, ys_test, 0.5)
plt.plot(k_history, loss_history, label='Training Loss', linestyle='-')
plt.plot(k_history, test_loss_history, label='Test Loss', linestyle='--')
plt.show()
print('test accuracy: ', model.accuracy(xs_test, ys_test))

g = torch.Generator().manual_seed(2147483647)
itos={i:s for s,i in model.stoi.items()}
for i in range(5):

    out = []
    current_bigram = [0,0]
    while True:
        bigram_tensor = torch.tensor([current_bigram], dtype=torch.long)
        logits = model.forward(bigram_tensor)
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))