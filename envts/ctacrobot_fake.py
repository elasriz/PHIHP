import gymnasium as gym
from gymnasium import core, spaces
from gymnasium.envs.registration import register
import numpy as np
from typing import Any, Callable, List, Optional, Set
from os import path

from gymnasium.envs.classic_control import utils

from typing import Optional

import numpy as np
from numpy import cos, pi, sin


from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding

from source.models import model_factory
from source.envs import Observer
import torch


from pathlib import Path

import os
import math


NO_RETURNS_RENDER = {"rgb_array"}

# list of modes with which render returns just a single frame of the current state
SINGLE_RENDER = {"single_rgb_array", "single_depth_array", "single_state_pixels"}


class CtAcrobotEnv(gym.Env):


    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 15}
    dt = 0.2

    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi


    friction_norm1 = 0.01
    friction_norm2 = 0.05


    torque_noise_max = 0.0

    SCREEN_DIM = 500
    

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    dt = 0.2

    

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None


    def __init__(self,
                 observer: object =Observer, 
                 render_mode: Optional[str] = "rgb_array",                  
                 model_class: str = "MLPModel_pendulum",
                 pointeur: int = 48, 
                 model_path: str = "models/{}_model_{}.tar",
                 directory: str = "training_model_acrobot/episode_10",
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

        self.render_mode = "rgb_array"
        self.screen = None
        self.clock = None
        self.isopen = True
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        low = -high
        
        self.observation_space = spaces.Box(shape=(6,), low=-high, high=high)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.state = None

        self.load_dynamics(model_class=self.model_class,
                               model_path=self.model_path,
                               **self.kwargs)
        
        self.seed()
        self.dt = 0.2

    @property
    def name(self):
        return self.dynamics_model.__class__.__name__

    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
        )
    
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






    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.1, 0.1  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
            np.float32
        )

        if self.render_mode == "human":
            self.render()
        return self._get_ob(), {}




        


    def seed(self, seed=None, options=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, a):


        s = self.state

        assert s is not None, "Call reset before using AcrobotEnv object."
        torque = a


        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max
            )

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        #s_augmented = np.append(s, torque)

        u = torch.tensor(torque, dtype=torch.float32).to(self.device)

        state = torch.tensor(self.state, dtype=torch.float32).to(self.device)
        state = state.expand(1, -1)

        ns = self.predict_transition(state, u)[0][0]

        ns[0] = wrap_to_pi(ns[0])
        ns[1] = wrap_to_pi(ns[1])
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)

        self.state = ns

        terminated = self._terminal()
        reward = -1.0 if not terminated else 0.0
        #reward = -cos(s[0]) - cos(s[1] + s[0])




        if self.render_mode == "human":
            self.render()
        return (self._get_ob(), reward, terminated, False, {})


    def _terminal(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0)
    

    def _dsdt(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        friction_norm1 = self.friction_norm1
        friction_norm2 = self.friction_norm2
        g = 9.8
        a = s_augmented[4:]
        s = s_augmented[:4]

        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]






        a0 =  - friction_norm1 * dtheta1
        a1 = a[0] - friction_norm2 * dtheta2
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )


        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a1 + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a1 + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
            ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 =  -(a0 + d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0, 0.0

    def render(self):
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
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        s = self.state

        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        p1 = [
            -self.LINK_LENGTH_1 * cos(s[0]) * scale,
            self.LINK_LENGTH_1 * sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]) * scale,
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1 * scale, self.LINK_LENGTH_2 * scale]

        pygame.draw.line(
            surf,
            start_pos=(-2.2 * scale + offset, 1 * scale + offset),
            end_pos=(2.2 * scale + offset, 1 * scale + offset),
            color=(0, 0, 0),
        )

        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))

            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def wrap_to_pi(x):

    return np.mod(x + np.pi, 2 * np.pi) - np.pi



def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    Args:
        x: scalar
        m: The lower bound
        M: The upper bound
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)




def rk4(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.
    Example for 2D system:
        >>> def derivs(x):
        ...     d1 =  x[0] + 2*x[1]
        ...     d2 =  -3*x[0] + 4*x[1]
        ...     return d1, d2
        >>> dt = 0.0005
        >>> t = np.arange(0.0, 2.0, dt)
        >>> y0 = (1,2)
        >>> yout = rk4(derivs, y0, t)
    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times
    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dt * k3))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    #print(yout[-1])
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1][:4]

def symplectic(derivs, y0, dt):
    
    y = np.asarray(derivs(y0))
    yout = y0[:4] + dt  * y[:4]
    yout[0] = y0[0] + dt  * yout[2]
    yout[1] = y0[1] + dt  * yout[3]

    return yout

register(
    id='im_ctacrobot',
    entry_point='envts.ctacrobot_fake:CtAcrobotEnv',
    max_episode_steps=500,
)
