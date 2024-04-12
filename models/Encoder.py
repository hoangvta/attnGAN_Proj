import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo
from torchvision import models

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def Conv1x1(in_channels, out_channels, bias=False):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=1, padding=0, bias=bias)


def Conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=1, bias=False)


class RNN_Encoder(nn.Module):
    def __init__(self, number_tokens, number_input=300, drop_chance=0.5,
                 number_hidden=128, number_layer=1, bidirectional=True):
        super(RNN_Encoder, self).__init__()
        self.n_step = 25
        self.number_tokens = number_tokens
        self.number_input = number_input
        self.drop_chance = drop_chance
        self.bidirectional = bidirectional
        if bidirectional:
            self.number_directions = 2
        else:
            self.number_directions = 1

        self.number_hidden = number_hidden // self.number_directions
        self.number_layer = number_layer

        self.define_modules()

    def define_modules(self):
        self.encoder = nn.Embedding(self.number_tokens, self.number_input)
        self.dropout = nn.Dropout(self.drop_chance)
        self.rnn = nn.LSTM(self.number_input, self.number_hidden,
                           self.number_layer, batch_first=True,
                           dropout=self.drop_chance,
                           bidirectional=self.bidirectional)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.number_layer * self.number_directions,
                                    bsz, self.number_hidden).zero_(),
                weight.new(self.number_layer * self.number_directions,
                                    bsz, self.number_hidden).zero_())

    def forward(self, captions, caption_length, hidden, mask=None):
        emb = self.dropout(self.encoder(captions))

        caption_length = caption_length.data.tolist()
        emb = pack_padded_sequence(emb, caption_length, batch_first=True, enforce_sorted=False)

        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]
        words_emb = output.transpose(1, 2)
        sentences_emb = hidden[0].transpose(0, 1).contiguous()
        sentences_emb = sentences_emb.view(-1, self.number_hidden * self.number_directions)

        return words_emb, sentences_emb


class CNN_Encoder(nn.Module):
    def __init__(self, nef):
        super(CNN_Encoder, self).__init__()
        self.nef = nef

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False

        self.define_modules(model)
        self.init_trainable_weights()

    def define_modules(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = Conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        self.emb_features.weight.data.uniform_(-0.1, 0.1)
        self.emb_cnn_code.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        x = self.Conv2d_1a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)  # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 64

        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)  # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 35 x 35 x 192
        x = self.Mixed_5b(x)  # 35 x 35 x 256
        x = self.Mixed_5c(x)  # 35 x 35 x 288
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 17 x 17 x 768
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        # image region features
        features = x  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 8 x 8 x 1280
        x = self.Mixed_7b(x)  # 8 x 8 x 2048
        x = self.Mixed_7c(x)  # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)  # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)  # 1 x 1 x 2048
        x = x.view(x.size(0), -1)  # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)  # 512

        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code
