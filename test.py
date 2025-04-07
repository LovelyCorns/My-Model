import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import load_model, save_model

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(100, 100)
        self.b = self.a

    def forward(self, x):
        return self.b(self.a(x))

model = BigramLanguageModel()
PATH = 'test_BERT.safetensors'
# PATH = 'test-BERT.pth'
# torch.save(model.state_dict(), PATH)
save_model(model=model, filename = PATH)
