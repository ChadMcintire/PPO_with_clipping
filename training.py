
def training_loop(env, args):

    #Create a model for PPO
    model = PPO(policy_class=FeedForwardNN, env, args)
