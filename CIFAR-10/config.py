## HYPER-PARAMETERS
import torch

CHN = [64,128,256]
LABEL = 10
EMBEDDING = 1024
IMSIZE = 32
NOISE = 256 ## 1024
GEN_CHN = [256,128,64,32]
GEN_EMBEDDING = 256
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 300
CRITIC_ITER = 5
LAMBDA_GP = 10 ## from paper
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')