from PPO import PPO as PPO

def training_loop(env, args):

    #Create a model for PPO
    model = PPO(env, args)

    if args.train and args.load_model:
        model.load_checkpoint(ckpt_path, evaluate=False)
        print("load existing model and restart training")
        
    elif args.train and not args.load_model:
        print("Begin fresh training")

    model.learn()

    
