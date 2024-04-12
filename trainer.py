import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from loss_functions import *

import config.settings as config

from torchvision.utils import save_image


def save_generate_img(generator, epoch, statics):
    fake_imgs = generator(*statics)[0]
    imgx64 = fake_imgs[0]
    imgx128 = fake_imgs[1]

    save_image(imgx64[0], f"x64/{epoch}.png")
    save_image(imgx128[0], f"x128/{epoch}.png")


def get_optimizer(generator, discriminators):
    d_optimizers = []
    for i in range(len(discriminators)):
        d_optimizers.append(optim.Adam(discriminators[i].parameters(),
                                       lr=config.LR,
                                       betas=(0.5, 0.999)))
    g_optimizer = optim.Adam(generator.parameters(),
                             lr=config.LR,
                             betas=(0.5, 0.999))
    return g_optimizer, d_optimizers


def train(cnn_model, rnn_model, generator, discriminators, data_loader, damsm_train=False):    
    generator.train()
    for i in range(len(discriminators)):
        discriminators[i].train()

    g_optim, d_optims = get_optimizer(generator, discriminators)
    
    real_labels = torch.ones((data_loader.batch_size,)).float().to(config.DEVICE)
    fake_labels = torch.zeros((data_loader.batch_size,)).float().to(config.DEVICE)
    labels = torch.LongTensor(list(range(data_loader.batch_size))).to(config.DEVICE)
    z_noise = config.Z_DIM

    statics = None
    
    for epoch in range(config.EPOCHS * 4):
        cur_time = time.time()
        for idx, (imgs, caps, lengths, ids) in enumerate(data_loader):
            batch_size = imgs.size(0)
            if batch_size < data_loader.batch_size: continue
                
            imgs, caps, lengths, ids = imgs.to(config.DEVICE), caps.to(config.DEVICE), lengths.to(config.DEVICE), ids.to(config.DEVICE)
            if damsm_train:
                DAMSM_iter(rnn_model, cnn_model, (imgs, caps, lengths, ids), transfer=False)
                
            real_imgs = [F.interpolate(imgs, 64), F.interpolate(imgs, 128)]
    
            
            noises = torch.FloatTensor(batch_size, z_noise).to(config.DEVICE)
    
            hidden = rnn_model.init_hidden(batch_size)
            
            word_embs, sentences_emb = rnn_model(caps.long(), lengths.long(), hidden)
            word_embs, sentences_emb = word_embs.detach(), sentences_emb.detach()
    
            mask = (caps == 0).to(config.DEVICE)
            num_words = word_embs.size(2)
    
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
    
            #save first iter
            noises.data.normal_(0, 1)
            if statics is None:
                statics = noises, sentences_emb, word_embs, mask
            
            total_d_loss = 0
            for _ in range(config.DISCRIMINATOR_REPEAT):
                for i in range(len(discriminators)):
                    d_noises = torch.randn((batch_size, z_noise)).to(config.DEVICE)
                    fake_imgs = generator(d_noises, sentences_emb, word_embs, mask)[0]
                    discriminators[i].zero_grad()
                    loss = discriminator_loss(discriminators[i], real_imgs[i], fake_imgs[i], sentences_emb, real_labels, fake_labels)
                    loss.backward()
                    d_optims[i].step()
                    
                    total_d_loss += loss
                    
            fake_imgs, _, mu, log_var = generator(noises, sentences_emb, word_embs, mask)
            # update generator
            generator.zero_grad()
            total_g_loss = generator_loss(discriminators, cnn_model, fake_imgs, real_labels, 
                                          word_embs, sentences_emb, labels, lengths, ids)
            total_g_loss += torch.mean(mu.pow_(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)).mul_(-0.5)
            total_g_loss.backward()
            g_optim.step()
            
        save_generate_img(generator, epoch, statics)
        print(f"Epoch {epoch} time {time.time() - cur_time}")
        print(f"g_loss: {total_g_loss}, d_loss: {total_d_loss}")
        
    return generator, discriminators, statics