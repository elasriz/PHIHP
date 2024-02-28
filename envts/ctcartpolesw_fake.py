"""
Continuous action version of the classic cart-pole system implemented by Rich Sutton et al.
"""
import gymnasium as gym
from gymnasium.envs.registration import register

from gymnasium.envs.classic_control import utils
from gymnasium.utils import seeding
from gymnasium.error import DependencyNotInstalled
from source.models import model_factory
from source.envs import Observer
import torch
import numpy as np
from typing import Any, Callable, List, Optional, Set
from os import path
import math
from pathlib import Path
import os
from math import cos, sin

NO_RETURNS_RENDER = {"human"}

# list of modes with which render returns just a single frame of the current state
SINGLE_RENDER = {"single_rgb_array", "single_depth_array", "single_state_pixels"}


class CartpoleEnv(gym.Env):
    def __init__(self,
                 observer: object =Observer, 
                 render_mode: Optional[str] = "rgb_array",                  
                 model_class: str = "MLPModel_pendulum",
                 pointeur: int = 48, 
                 model_path: str = "models/{}_model_{}.tar",
                 directory: str = "training_model_cartpolesw/episode_10",
                 **kwargs: dict) -> None:
        


        self.device = "cuda"
        self.observer = observer
        self.model_class = model_class
        self.pointeur = pointeur
        self.model_path = model_path
        self.directory = Path(directory)
        self.kwargs = kwargs
        self.dynamics_model = None
        self.action_repeat = 1


        self.min_action = -1.0
        self.max_action = 1.0
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_max = 10.0
        self.tau = 0.02  # seconds between state updates
        self.dt = 0.02




        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array(
            [
                np.inf,
                np.finfo(np.double).max,
                1.0,
                1.0,
                np.finfo(np.double).max,
            ],
            dtype=np.float32,
        )




        self.action_space = gym.spaces.Box(self.min_action, self.max_action, shape=(1,))
        self.observation_space = gym.spaces.Box(-high, high)

        self.load_dynamics(model_class=self.model_class,
                               model_path=self.model_path,
                               **self.kwargs)
        
        self.seed()
        self.screen = None
        self.screen_width=600
        self.screen_height=300
        self.clock=None
        self.state = None

        self.steps_beyond_terminated = None

    @property
    def name(self):
        return self.dynamics_model.__class__.__name__


    def load_dynamics(self, model_class, model_path, **kwargs):
        kwargs["action_size"] = 1

        kwargs["state_size"] = 4
        self.dynamics_model = model_factory(model_class, kwargs)

        path = os.path.join(self.directory, model_path.format(self.dynamics_model.__class__.__name__, self.pointeur))

        self.dynamics_model.load_state_dict(torch.load(path))
        print(f"Loaded {model_class} from {path}.")
        self.dynamics_model.eval()
        self.dynamics_model.to(self.device)
        return self

    def predict_transition(self, state, action):
        action = torch.repeat_interleave(action, self.action_repeat, dim=0)
        action = action.unsqueeze(0)
        
        new_state = self.dynamics_model.integrate(state, action)
        return new_state[::self.action_repeat, ...].cpu().numpy()



    def seed(self, seed=None, options=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):       



        
        
        x, x_dot, theta, theta_dot = self.state


        force = min(max(action[0], -1.0), 1.0) 
        u = torch.tensor([force], dtype=torch.float32).to(self.device)

        state = torch.tensor(self.state, dtype=torch.float32).to(self.device)
        state = state.expand(1, -1)
        
        x, x_dot, theta, theta_dot  = self.predict_transition(state, u)[0][0]
      
        self.state = (x, x_dot, angle_normalize(theta), theta_dot)
        x, x_dot, theta, theta_dot = self.state
        sintheta = np.sin(angle_normalize(theta))
        costheta = np.cos(angle_normalize(theta))
        reward = np.exp(-(x + self.length*sintheta)**2 - (self.length*costheta)**2 ) 


        return self._get_ob(), reward, False, False, {}

    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return np.array(
            [s[0], s[1], cos(s[2]), sin(s[2]), s[3]], dtype=np.float32
        )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.state[2] =  angle_normalize(np.pi + self.state[2])
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return  self._get_ob(), {}
    

    def get_random_state(self):
        return np.array([np.random.uniform(-self.x_threshold, self.x_threshold), \
                  np.random.uniform(-self.force_max, self.force_max), \
                  np.random.uniform(-self.theta_threshold_radians, self.theta_threshold_radians), \
                  np.random.uniform(-self.force_max, self.force_max)])
        
    def render(self, mode="rgb_array"):
        self.render_mode=mode
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

register(
    id='im_cartpolesw',
    entry_point='envts.ctcartpolesw_fake:CartpoleEnv',
    max_episode_steps=500,
)