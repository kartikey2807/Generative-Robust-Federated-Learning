import torch 

INPUT = 2
LABEL = 2
HIDDEN = [32,32]
EMBEDDING = 2
Z_DIM = 4
GENDIM = [8,4,4]
LEARNING_RATE = 0.00001
BATCH_SIZE = 256
EPOCHS = 30000
CRITIC_ITERS = 5
LAMBDA_GP = 10 ## given in WGAN-GP paper
IMP = 0.25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')