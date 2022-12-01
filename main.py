import argparse
import gym
from training import training_loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #float values
    parser.add_argument("--lr", default=0.005, help="model learing rate (default=.0001)")
    parser.add_argument("--gamma", default=0.95, help="discount factor (default=.99)")
    parser.add_argument("--clip", default=0.2, help="helps define the threshold to clip the ratio during SGA (default=.2)")
    parser.add_argument("--lr_actor", default=.005, help="learning rate for actor(default=0.0003)")
    parser.add_argument("--lr_critic", default=.005, help="learning rate for critic(default=0.001)")

    #int values
    #parser.add_argument("--save_model_freq", default=int(1e5), type=int, help="How often to save the model in steps (default=100000)")
    parser.add_argument("--save_model_freq", default=50, type=int, help="How often to save the model in steps (default=100000)")
    parser.add_argument("--timesteps_per_batch", default=4800, type=int, help="amount of steps for the roullout(default=4800)")
    parser.add_argument("--total_time_steps", default=200000000, type=int, help="number of steps to learn(default=200,000,000)")
    parser.add_argument("--max_timesteps_per_episode", default=1600, type=int, help="max amount of steps before a new episode will start (default=2048)")
    parser.add_argument("--n_updates_per_iteration", default=5, type=int, help="max amount of steps to do backprop on the actor critic model for each rollout (default=5)")
    parser.add_argument("--render_every_i", default=10, type=int, help="how often to display the environment (default=10)")
    parser.add_argument("--random_seed", default=0, help="set the random seed if required 0 = no random seed (default=0)")

    #string values
    parser.add_argument("--env_name", default="Pendulum-v1", help="Name of environment we plan to train on")
    parser.add_argument("--ckpt_path", default="checkpoints", help="default value for where to store or retrieve the path to the trained model")    

    #bool values 
    parser.add_argument('--train', dest="train", action='store_true', help='train a new model')
    parser.add_argument('--no-train', dest="train", action='store_false', help='evalate a previously trained model and run it')
    parser.set_defaults(train=True)

    parser.add_argument('--load_model', dest="load_model", action='store_true', help='load a model during training')
    parser.add_argument('--no-load_model', dest="load_model", action='store_false', help='Train a new model and do not load a previous model')
    parser.set_defaults(load_model=False)

    parser.add_argument('--render', dest="render", action='store_true', help='display what is happening in the environment (default=True)')
    parser.add_argument('--no-render', dest="render", action='store_false', help='do not display environment')
    parser.set_defaults(render=False)
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Env: {args.env_name}, Seed: {args.random_seed}, Trainging: {args.train}")
    print("---------------------------------------")


    # switch between the human readable mode and the mode we have to render.
    env = gym.make(args.env_name, render_mode="rgb_array")
    #env = gym.make(args.env_name, render_mode="human")

    if args.train:
        training_loop(env, args)
