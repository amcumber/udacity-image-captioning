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
    """
    Test Caption Model for the Decoding RNN portion of for the image captioning
    project
    """

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
            number of embeddings (standard length of input vector to lstm)
        hidden_size : int
            size of hidden layer
        vocab_size : int
            size of vocabulary (output of Encoder)
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
        batch_size = features.size(0)
        print(features.shape)  # 10, 256
        x_embed = self.embedding(captions)  # 10, 14, 256
        x = torch.cat((features.unsqueeze(dim=1), x_embed), dim=1)
        seq_size = x.size(1)

        x = x.contiguous().view(batch_size, seq_size, -1)
        hidden = self.init_hidden(batch_size)

        x, hidden = self.lstm(x, hidden)
        x = x.view(len(features), -1)
        # x = self.drop(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    # CITATION: Udacity Computer Vision - LSTM notebook + project init
    def init_hidden(self, batch_size):
        """
        At the start of training, we need to initialize a hidden state; there
        will be none because the hidden state is formed based on perviously
        seen data.  So, this function defines a hidden state with all zeroes
        and of a specified size.
        """
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (
            torch.randn(self.num_layers, batch_size, self.hidden_size),
            torch.randn(self.num_layers, batch_size, self.hidden_size),
        )

    def sample(self, inputs, states=None, max_len=20):
        """
        accepts pre-processed image tensor (inputs) and returns predicted
        sentence (list of tensor ids of length max_len)
        """
        pass
