import torch.cuda

EPOCHS = 50
LR = 2e-4
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 299

DF_DIM = 64
GF_DIM = 128
CONDITION_DIM = 100
Z_DIM = 100
R_NUM = 2

EMBEDDING_DIM = 256
CAPTIONS_PER_IMAGE = 10
WORD_SIZE = 5428
SENTENCE_SIZE = 66
IMAGE_NUMBER = 8190

LAMBDA = 1.0

GAMMA1 = 4
GAMMA2 = 5
GAMMA3 = 10

DISCRIMINATOR_REPEAT = 3
RNN_GRAD = 0.25
BIDIRECTION = True

DATA_DIR = 'flower-102'