from model import FeedForwardNN 
import torch
from matplotlib import pyplot as plt
from torch.distributions import MultivariateNormal


def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.
			Parameters:
				None
			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, render):

    while True:
        obs = env.reset()[0]
        print(type(obs))


        done = False

        #time steps so far
        t = 0

        ep_len = 0
        ep_ret = 0

        while not done:
            t += 1

            if render:
                image =  env.render()
                plt.figimage(image)
                plt.draw()
                plt.pause(0.00001)

                # the display gets incredibly slow if you do not clear the plot occasionally, i do every 10 images
                plt.clf()

            #mean = policy(obs).detach().numpy()
            action = policy(obs).detach().numpy()
            #cov_var = torch.full(size=(env.action_space.shape[0],), fill_value=0.5)
            #cov_mat = torch.diag(cov_var)


            #dist = MultivariateNormal(mean, cov_mat)

            # Sample an action from the distribution
            #action = dist.sample()


            obs, reward, done, truncated, info = env.step(action)

            ep_ret += reward

        ep_len = t

        plt.close('all')

        yield ep_len, ep_ret





def run(env, args):
    print("beginning runs")

    # Extract dimension of observation and action space
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    checkpoint = torch.load(args.ckpt_path)
    print(checkpoint)

    # Call the feed forward network so we don't carry the whole model with us
    policy = FeedForwardNN(obs_dim, act_dim)

    #load the pretrained weights into the model
    policy.load_state_dict(checkpoint['actor_state_dict'])

    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, args.render)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)

        
