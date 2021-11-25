import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    """Decoding RNN for project"""

    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers=1,
        p_drop=0.2,
    ):
        """Constructor

        Parameters
        ----------
        embed_size : int
            number of embeddings (output of Encoder)
        hidden_size : int
            size of hidden layer
        vocab_size : int
            size of vocabulary
        num_layers : int (default=1)
            number of lstm layers
        """
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.p_drop = p_drop
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=p_drop,
        )
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Distribute weights
        self.fc.weight.data.uniform_(-1, 1)
        self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, features, captions):
        # CITATION: Udacity Computer Vision - LSTM notebook
        x_embed = self.embedding(captions)
        x = torch.cat((features, x_embed))
        x = x.view(len(features), x_embed.shape[1], -1)
        hidden = self.init_hidden()

        x, hidden = self.lstm(x, hidden)
        x = x.view(len(features), -1)
        # x = self.drop(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    # CITATION: Udacity Computer Vision - LSTM notebook + project init
    def init_hidden(self):
        """At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on perviously seen data.
        So, this function defines a hidden state with all zeroes and of a specified size."""
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        # hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
        return (
            torch.randn(1, 1, self.hidden_size),
            torch.randn(1, 1, self.hidden_size),
        )

    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        pass
