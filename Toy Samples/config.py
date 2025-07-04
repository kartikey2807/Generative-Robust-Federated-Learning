import torch 

INPUT = 2
LABEL = 2
HIDDEN = [8,4]
EMBEDDING = 2
Z_DIM = 4
GENDIM = [4,2]
LEARNING_RATE = 0.00004
BATCH_SIZE = 256
EPOCHS = 21000
IMP = 0.02
CRITIC_ITERS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')