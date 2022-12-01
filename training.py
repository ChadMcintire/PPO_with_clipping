from PPO import PPO as PPO
import numpy as np
import torch
import random

def training_loop(env, args):

    #Create a model for PPO
    model = PPO(env, args)

    if args.train and args.load_model:
        model.load_checkpoint(ckpt_path, evaluate=False)
        print("load existing model and restart training")
        
    elif args.train and not args.load_model:
        print("Begin fresh training")

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)



    model.learn()

    
