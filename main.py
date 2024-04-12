from models.Encoder import RNN_Encoder, CNN_Encoder
from models.Generator import GenerativeNetwork
from models.Discriminator import DiscriminatorNetwork
from dataset import *
import config.settings as config

import torch
import torch.nn as nn

from torchvision import transforms


def create_models():
    def weight_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.orthogonal_(m.weight.data, 1.0)
        elif class_name.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif class_name.find('Linear') != -1:
            nn.init.orthogonal_(m.weight.data, 1.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    rnn_model = RNN_Encoder(config.WORD_SIZE, number_hidden=config.EMBEDDING_DIM)
    cnn_model = CNN_Encoder(config.EMBEDDING_DIM)

    gen_model = GenerativeNetwork()
    dis_models = [DiscriminatorNetwork(), DiscriminatorNetwork(down_sample_count=1)]

    gen_model.apply(weight_init)
    dis_models[0].apply(weight_init)
    dis_models[1].apply(weight_init)

    return rnn_model, cnn_model, gen_model, dis_models


def load_models(rnn_model, cnn_model, gen_model, dis_models,
                rnn_link, cnn_link, gen_link, dis_links):
    rnn_model.load_state_dict(torch.load(rnn_link))
    cnn_model.load_state_dict(torch.load(cnn_link))
    gen_model.load_state_dict(torch.load(gen_link))
    for i in range(len(dis_models)):
        dis_models[i].load_state_dict(torch.load(dis_links[i]))

    return rnn_model, cnn_model, gen_model, dis_models


def create_data_loader():
    imsize = config.IMAGE_SIZE
    transform = transforms.Compose([
        transforms.Resize(int(imsize)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    return DataLoader(TextDataset('.', transform=transform),
                      batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    pass