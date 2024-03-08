import argparse

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import random


def display_video(frames, path="episode.mp4", framerate=30):
    if not frames:
        return
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=framerate, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(path, writer=writer)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('True', 'yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def wrap_to_pi(x):
    if isinstance(x, np.ndarray):
        return np.mod(x + np.pi, 2 * np.pi) - np.pi
    else:
        return torch.fmod(x + np.pi, 2 * np.pi) - np.pi


def rk4_step_func(func, t, dt, y, k1=None):
    if k1 is None:
        k1 = func(t, y)
    half_dt = dt * 0.5
    k2 = func(t + half_dt, y + half_dt * k1)
    k3 = func(t + half_dt, y + half_dt * k2)
    k4 = func(t + dt, y + dt * k3)
    return (k1 + 2 * (k2 + k3) + k4) / 6


def rk4_step_func_autonomous(func, dt, y, k1=None):
    if k1 is None:
        k1 = func(y)
    half_dt = dt * 0.5
    k2 = func(y + half_dt * k1)
    k3 = func(y + half_dt * k2)
    k4 = func(y + dt * k3)
    return (k1 + 2 * (k2 + k3) + k4) / 6


def seed_all(seed, env=None):
    if seed is None:
        seed = np.random.randint(2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if env is not None:
        env.unwrapped.seed(seed)
        env.action_space.seed(seed)
    return seed


	

	