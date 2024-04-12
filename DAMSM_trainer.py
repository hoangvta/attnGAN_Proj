import time
import torch
import config.settings as config

import torch.optim as optim

from loss_functions import sent_loss, word_loss


def DAMSM_iter(rnn_model, cnn_model, data, transfer=True):
    rnn_model.train()
    cnn_model.train()
    
    img, captions, lengths, idx_ids = data

    if transfer:
        img = img.to(config.DEVICE)
        captions = captions.to(config.DEVICE)
        lengths = lengths.to(config.DEVICE)
        idx_ids = idx_ids.to(config.DEVICE)

    batch_size = img.size(0)
    labels = torch.LongTensor(list(range(batch_size))).to(config.DEVICE)

    features, cnn_code = cnn_model(img)

    hidden = rnn_model.init_hidden(batch_size)

    word_embs, sentence_emb = rnn_model(captions.long(), lengths.long(), hidden)

    w_loss0, w_loss1, attn_maps = word_loss(features, word_embs, lengths, labels, idx_ids, batch_size)

    loss = w_loss0 + w_loss1
    s_loss0, s_loss1 = sent_loss(cnn_code, sentence_emb, labels, idx_ids, batch_size)
    loss += s_loss0 + s_loss1

    loss.backward()

    torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), config.RNN_GRAD)

    optimizer.step()

    rnn_model.eval()
    cnn_model.eval()


def DAMSM_train_epoch(rnn_model, cnn_model, data_loader, save_dir: tuple[str, str] = ("rnn_model_state_dict.pt", "cnn_model_state_dict.pt")):
    rnn_model, cnn_model = load_models(rnn_model, cnn_model, save_dir)
    rc_optimizer = get_optimizer(rnn_model, cnn_model)
    
    
    for epoch in range(config.EPOCHS):
        cur_time = time.time()

        for idx, data in enumerate(data_loader):
            DAMSM_iter(rnn_model, cnn_model, data)
            
            if idx % 30 == 0:
                print(f"{epoch}:{idx}: {loss}")
                
        print(f"Epoch time: {time.time() - cur_time : .4f}")

        
    save_models(rnn_model, cnn_model, save_dir)


def save_models(rnn_model, cnn_model, save_dir):
    if save_dir is None:
        return

    torch.save(rnn_model.state_dict(), save_dir[0])
    torch.save(cnn_model.state_dict(), save_dir[1])


def load_models(rnn_model, cnn_model, save_dir):
    try:
        rnn_model.load_state_dict(torch.load(save_dir[0]))
        cnn_model.load_state_dict(torch.load(save_dir[1]))
    finally:
        return rnn_model, cnn_model


def get_rc_optimizer(rnn_model, cnn_model):
    para = get_parameters(rnn_model, cnn_model)
    optimizer = optim.Adam(para, lr=config.LR, betas=(0.5, 0.999))

    return optimizer


def get_parameters(rnn_model, cnn_model):
    para = list(rnn_model.parameters())
    for p in cnn_model.parameters():
        if p.requires_grad:
            para.append(p)

    return para
