import numpy as np

from source.envs import plot_trajectories


class RandomPolicy(object):
    name: str = "Random"

    def __init__(self, env: object, noise_tau: float = 20, sigma: float = 0.1, noise_schedule_tau=50, **kwargs: dict) -> None:
        self.env = env
        self.name = "Random"
        self.action_space = env.action_space
        self.noise_tau = noise_tau
        self.sigma = sigma
        self.time = 0
        self.noise_schedule_tau = noise_schedule_tau
        self.last_action = None

    def ornstein_uhlenbeck_noise(self):

        low = getattr(self.action_space, "minimum", getattr(self.action_space, "low", None))
        high = getattr(self.action_space, "maximum", getattr(self.action_space, "high", None))

        noise = np.random.uniform(low=low, high=high)\
            .astype(self.action_space.dtype, copy=False)
        return self.last_action - 1 / self.noise_tau * self.last_action + self.sigma * noise \
            if self.last_action is not None else self.sigma * noise


    def act(self, best_action=None):
        action = self.ornstein_uhlenbeck_noise()
        self.last_action = action
        if best_action is not None:
            action = self.schedule * best_action + (1 - self.schedule) * action
        return action

    def step(self):
        self.time += 1

    def plot(self, dataframe, env_name):
        axes = None

        # True trajectory
        df = dataframe
        if env_name == 'ctcartpole':
            df_states = np.stack([df["state_cart_position"], df["state_cart_velocity"], df["state_pole_angle"], df["state_pole_velocity"]], axis=1)
            df_actions = np.stack([df["action_0"]], axis=1)
        elif env_name == 'ctacrobot':
            df_states = np.stack([df["state_angle_1"], df["state_angle_2"], df["state_angular_vel_1"], df["state_angular_vel_2"]], axis=1)
            df_actions = np.stack([df["action_0"], df["action_1"]], axis=1)
        elif env_name == 'pendulum':
            df_states = np.stack([df["state_angle"], df["state_angular_vel"]], axis=1)
            df_actions = np.stack([df["action_0"]], axis=1)

        axes = plot_trajectories(df["time"], df_states, df_actions, self.env.action_space,
                                 plot_args={"linestyle": '-', "linewidth": 2}, axes=axes)



        
    @property
    def schedule(self):
        return min(self.time / self.noise_schedule_tau, 1)