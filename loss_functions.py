import torch
import torch.nn as nn
import config.settings as config


def func_attn(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def word_loss(img_features, word_embs, cap_len, labels, class_ids, batch_size):
    masks = []
    attn_maps = []
    similarities = []
    cap_len = cap_len.data.tolist()

    for i in range(batch_size):
        mask = (class_ids == class_ids[i])
        mask[i] = 0
        masks.append(mask.reshape((1, -1)))

        word_num = cap_len[i]
        word = word_embs[i, :, :word_num].unsqueeze(0).contiguous()
        word = word.repeat(batch_size, 1, 1)
        context = img_features

        weighted_context, attn = func_attn(word, context, config.GAMMA1)
        attn_maps.append(attn[i].unsqueeze(0).contiguous())

        word = word.transpose(1, 2).contiguous()
        weighted_context = weighted_context.transpose(1, 2).contiguous()

        word = word.view(batch_size * word_num, -1)
        weighted_context = weighted_context.view(batch_size * word_num, -1)

        row_similarity = cosine_similarity(word, weighted_context)
        row_similarity = row_similarity.view(batch_size, word_num)

        row_similarity.mul_(config.GAMMA2).exp_()
        row_similarity = row_similarity.sum(dim=1, keepdim=True)
        row_similarity = torch.log(row_similarity)

        similarities.append(row_similarity)

    similarities = torch.cat(similarities, dim=1)
    masks = torch.cat(masks, 0)

    similarities.data.masked_fill_(masks, float('-inf'))

    similarities = similarities * config.GAMMA3
    similarities_t = similarities.transpose(0, 1)

    loss = nn.CrossEntropyLoss()(similarities, labels)
    loss_t = nn.CrossEntropyLoss()(similarities_t, labels)
    return loss, loss_t, attn_maps


def sent_loss(cnn_code, rnn_code, labels, idx_ids, batch_size, eps=1e-8):
    masks = []

    for i in range(batch_size):
        mask = (idx_ids == idx_ids[i])
        mask[i] = 0
        masks.append(mask.reshape((1, -1)))

    masks = torch.cat(masks, 0)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)
    # why un-squeezed?

    cnn_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    score = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm = torch.bmm(cnn_norm, rnn_norm.transpose(1, 2))
    score = score / norm.clamp(min=eps) * config.GAMMA3
    score = score.squeeze()

    # how to run with batch_size = 1
    score.data.masked_fill_(masks, float('-inf'))

    score_t = score.transpose(0, 1)

    loss = nn.CrossEntropyLoss()(score, labels)
    loss_t = nn.CrossEntropyLoss()(score_t, labels)

    return loss, loss_t


def discriminator_loss(netD, real_imgs, fake_imgs, conditions,
                       real_labels, fake_labels):

    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())

    # loss
    cond_real_logits = netD.conditional_discriminator(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.conditional_discriminator(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.conditional_discriminator(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.unconditional_discriminator is not None:
        real_logits = netD.unconditional_discriminator(real_features)
        fake_logits = netD.unconditional_discriminator(fake_features)

        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)

        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.

    return errD


def generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids):
    numDs = len(netsD)
    batch_size = real_labels.size(0)

    errG_total = 0

    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].conditional_discriminator(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].unconditional_discriminator is not None:
            logits = netsD[i].unconditional_discriminator(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss

        # Ranking loss
        if i == (numDs - 1):
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = word_loss(region_features, words_embs,
                                            cap_lens, match_labels,
                                            class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * config.LAMBDA

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                         match_labels, class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * config.LAMBDA

            errG_total += w_loss + s_loss

    return errG_total
