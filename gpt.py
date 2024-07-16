import torch
import numpy
import model

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

model = model.GPTLanguageModel(vocab_size=vocab_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
m = model.to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Prompt the user to enter the fixed-size context
terminal_context = input("Enter the fixed-size context: ")
context = torch.tensor([encode(terminal_context)], dtype=torch.long, device=device)

# generate from the model
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
