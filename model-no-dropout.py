from typing import List
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
        p_drop=0,
        device="cpu",
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
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Distribute weights
        # self.fc.weight.data.uniform_(-1, 1)
        # self.embedding.weight.data.uniform_(-1, 1)
        self.device = device
        if device is not "cpu":
            self.to(device)

    def to(self, device):
        super().to(device)
        self.device = device

    def forward(self, features, captions):
        # CITATION: Udacity Computer Vision - LSTM notebook
        batch_size = features.size(0)
        x_embed = self.embedding(captions)  # 10, 14, 256
        # remove <end> tag
        x_embed = x_embed[:, :-1, :]
        x = torch.cat((features.unsqueeze(dim=1), x_embed), dim=1)
        seq_size = x.size(1)

        x = x.contiguous().view(batch_size, seq_size, -1)
        hidden = self.init_hidden(batch_size)

        x, hidden = self.lstm(x, hidden)
        # x = x.contiguous().view(batch_size, -1, self.hidden_size)
        # x = self.drop(x)
        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
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
            torch.randn(
                self.num_layers,
                batch_size,
                self.hidden_size,
            ).to(self.device),
            torch.randn(
                self.num_layers,
                batch_size,
                self.hidden_size,
            ).to(self.device),
        )

    def sample(self, inputs, states=None, max_len=20) -> List[int]:
        """
        accepts pre-processed image tensor (inputs) and returns predicted
        sentence (list of tensor ids of length max_len)
        """
        # CITATION: Udacity Computer Vision - LSTM notebook
        # input_size = inputs.size(2)  # 1,1,256
        # seed_seq = torch.zeros(batch_size, max_len - 1, input_size)
        # seed_seq = seed_seq.to(self.device)
        # input_w_seed = torch.cat((inputs, seed_seq), dim=1)
        # x = input_w_seed.contiguous().view(1, max_len, -1)
        # hidden = self.init_hidden(batch_size)
        output = []
        batch_size = 1
        hidden = self.init_hidden(batch_size)
        # hidden = None
        x = inputs
        for _ in range(max_len):
            x, hidden = self.lstm(x, hidden)
            x = self.fc(x)
            word_idx = torch.argmax(x, dim=2)
            output.append(word_idx.item())
            x = self.embedding(word_idx)

        # x = F.log_softmax(x, dim=1)
        # output = x.type(torch.IntTensor).squeeze().tolist()
        return output
