import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.registration import register
import numpy as np
from typing import Any, Callable, List, Optional, Set
from os import path
import os

from source.models import model_factory
from source.envs import Observer
import torch

from pathlib import Path

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

NO_RETURNS_RENDER = {"human"}

# list of modes with which render returns just a single frame of the current state
SINGLE_RENDER = {"single_rgb_array", "single_depth_array", "single_state_pixels"}


class GymPendulum(gym.Env):

    metadata = {'render_modes': ['human', 'rgb_array']}
    def __init__(self, 
                 observer: object =Observer, 
                 render_mode: Optional[str] = "rgb_array",                  
                 model_class: str = "MLPModel_pendulum",
                 pointeur: int = 48, 
                 model_path: str = "models/{}_model_{}.tar",
                 directory: str = "training_model_pendulum/episode_10",
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

        


        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05


        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True


        high = np.array([np.pi, self.max_speed], dtype=np.float32)


        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

        self.load_dynamics(model_class=self.model_class,
                               model_path=self.model_path,
                               **self.kwargs)

    @property
    def name(self):
        return self.dynamics_model.__class__.__name__

    def load_dynamics(self, model_class, model_path, **kwargs):
        kwargs["action_size"] = 1
        #print (kwargs["action_size"])
        kwargs["state_size"] = 2
        self.dynamics_model = model_factory(model_class, kwargs)

        #path = self.directory / model_path.format(self.dynamics_model.__class__.__name__, self.pointeur)
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



    def step(self, u):
        #u = u[0]
        th, thdot = self.state  


        u = np.clip(u, -1.0, 1.0)

        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        u = torch.tensor(u, dtype=torch.float32).to(self.device)
        state = torch.tensor(self.state, dtype=torch.float32).to(self.device)
        state = state.expand(1, -1)
        newth, newthdot  = self.predict_transition(state, u)[0][0]
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)


        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = gym.utils.verify_number_and_cast(x)
            y = gym.utils.verify_number_and_cast(y)
            high = np.array([x, y])

        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None


        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(self.state, dtype =np.float32)
        #theta, thetadot = self.state
        #return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
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
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def seed(self, seed):
        self.seed=seed

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


register(
    id='im_pendulum',
    entry_point='envts.pendulum_fake:GymPendulum',
    max_episode_steps=200,
)