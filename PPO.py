from torch.distributions import MultivariateNormal


class PPO:
    """
		PPO with clipping basically the baseline from OpenAI
	"""
    def __init__(self, env, args):
        assert(type(env.observation_space) == gym.space.Box)
        assert(type(env.action_space) == gym.space.Box)

        # setup environment and dimensions you will need.
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # Initialize actor and critic                                                       #Algorithm step 1
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        # set up an optimizer for the actor and critic
        self.optimizer = torch.optim.Adam([
                            {'params': self.actor.parameters(), 'lr': args.lr_actor},
                            {'params': self.critic.parameters(), 'lr': args.lr_critic},
        ])

        #Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.total_time_steps = args.total_time_steps

    def learn(self):
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < self.total_time_steps:                                              #Algorithm Step 2
            #collect batch simulation
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  #Algorithm Step 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Calculate the advantage at k-th iteration
            V, _ = self.evaluate
            A_k = batch_rtgs - V.detach()

            # One of the only tricks I used isn't in the pseudocode. Normalizing advantages isn't
            # in theoretically necessary, but in practice it decreases the variance of our 
            # advantages and make convergencce much more stable and faster. I added this because
            # solving some environments was too unstable without it
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                #Calculate V_phi and pi_theta(a_t| s_t)
                V, current_log_probs = self.evaluate(batch_ops, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # We just subtract the logs, which is the same as
                # dividing the values and then canceling the logs with e^logs.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(current_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize




    def rollout(self):
        """
            This is where we collect the batch of data from simulation. Since this is an on-policy algorithm, we'll
            need to collect a fresh batch of data each time we iterate the actor/critic networks.

            
            returns:
                batch_obs: the observations collected this batch. Shape (number of timesteps, dimension of observation)
                batch_acts: the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs: the log probabilities of each action taken this batch. Shape (number of timesteps)
                batch_rtgs: the Reward-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens: the lengths of each episode this batch. Shape: (number of episodes)

        """
        
        # batch return of the data after the rollout
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # keep track of the episodic reward, clears on new episode
        ep_rew = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # clear out out rewards

            # Reset the environment
            obs = self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if self.render and self.render_every_i == 0 and len(batch_lens) == 0:
                    self.env.render()

                t += 1 # Increment observations ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env.
                action, log_prob = self.get_action(obs)
                obs, reward, done, _ = self.env.step(action)

                # Track recent reward, action, and action log probabilities
                ep_rews.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_probs)

                #If the episode is in a terminal state, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        # Reshape data as tensors in the shape specified in function description, before returning 
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Log the episodic returns and episodic lengths in the batch.

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

                



    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout

            Parameters:
                obs: the observations at the current timesteps

            Return:
                action: the action to take, as a numpy array
                log_prob: the log probability of the selected action in the distribution

        """

        # Query the actor for a mean action
        mean = self.actor(obs)

        # Create a distribution with the mean action and std from the covariate matrix above. 
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of the action in our distribution
        return action.detach().numpy(), log_prob.detach()
        

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of 
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameter: 
                batch_obs: the observations from the most recently collected batch as a tensor.
                           Shape: (number of timesteps in batch, dimension of observations).

                batch_acts: the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)

            
            Return:
                V: the predicted values of batch_obs
                log_probs: the log probabilities of the actions taken in batch_acts given batch_obs
        """

        # Query critic network for a value V for each batch_obs. Shape of V should be the same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network 
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = Multivariate(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs


    
    def save_checkpoint(self,  suffix="", ckpt_path="checkpoints"):
            ch_pt = "check_points/"

            if not os.path.exists(ch_pt + ckpt_path + "/"):
                os.makedirs(ch_pt + ckpt_path + "/")
            torch.save({"q_net_state_dict": self.q.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()}, ch_pt + ckpt_path + "/" + name)


    def load_checkpoint(self, ckpt_path, evaluate=True):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint['q_net_state_dict'])
            self.critic.load_state_dict(checkpoint["q_net_state_target_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if evaluate:
                self.q.eval()
                self.q_target.eval()

            else:
                self.q.train()
                self.q_target.train()
