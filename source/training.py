from pathlib import Path
import pandas as pd
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
import logging

from source.models import DynamicsModel


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self,
                 model: DynamicsModel,
                 device: str = "cuda",
                 epochs: int = 1000,
                 lr: float = 0.01,
                 lambda_0: float = 1,
                 Niter: int = 1,
                 aph_tau: int = 10,
                 data_path: str = "data/dataset.csv",
                 model_path: str = "models/{}_model_{}.tar",
                 directory: str = "out/",
                 seed: int = 5,
                 **kwargs: dict) -> None:
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.data_path = data_path
        self.model_path = model_path
        self.directory = Path(directory)
        self.data = None
        self.train_data, self.test_data = None, None
        self.loss = torch.nn.MSELoss()
        self.losses = []
        self.lambda_0 = lambda_0
        self.Niter = Niter
        self.aph_tau = aph_tau
        self.seed = seed



    def load_data(self, df=None):
        if df is None:
            print(f"Loading data from {self.directory / self.data_path}")
            df = pd.read_csv(self.directory / self.data_path)



        states = df.filter(like='state').to_numpy()
        actions = df.filter(like='action').to_numpy()
        time = df["time"].to_numpy()


        # Transitions
        next_states = states[1:]
        time = time[:-1]
        states = states[:-1]
        actions = actions[:-1]

        # Remove env resets
        no_resets = np.where(np.diff(time) > 0)
        next_states, time, states, actions = next_states[no_resets], time[no_resets], states[no_resets], actions[no_resets]

        # To tensors
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        time = torch.tensor(time, dtype=torch.float).to(self.device)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        self.data = (time, states, actions, next_states)

    def split_train_test(self, ratio=0.8):
        time, states, actions, next_states = self.data
        ind_list = np.arange(time.shape[0]).tolist()
        np.random.shuffle(ind_list)
        train_size = int(states.shape[0] * ratio)
        train_data = (time[ind_list[:train_size]], states[ind_list[:train_size]], actions[ind_list[:train_size]], next_states[ind_list[:train_size]])
        test_data = (time[ind_list[train_size:]], states[ind_list[train_size:]], actions[ind_list[train_size:]], next_states[ind_list[train_size:]])
        self.train_data, self.test_data = train_data, test_data

    def save_model(self):

        out_path = self.directory / self.model_path.format(self.model.__class__.__name__, self.seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saved model to {out_path}")
        torch.save(self.model.state_dict(), out_path)

    def compute_loss(self, data):
        times, states, actions, next_states = data
        predictions = self.model(states, actions)

        return torch.mean(torch.sum((predictions - next_states) ** 2, dim=1))

    def compute_loss_aphynity(self, data, lambda_t):
        _, states, actions, next_states = data
        predictions = self.model(states, actions)
        lpred = self.loss(predictions, next_states)
        Fa, Fp = self.model.get_Fa()
        norm_Fa_ = torch.norm(Fa) / torch.norm(Fp)
        norm_Fa = norm_Fa_ #########################
        loss = lpred + norm_Fa / lambda_t
        return loss, lpred, norm_Fa

    def train(self, df=None):
        self.load_data(df)
        self.split_train_test()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.losses = np.full((self.epochs, 2), np.nan)
        epochs = tqdm.trange(self.epochs, desc="Train dynamics") if logger.getEffectiveLevel() == logging.DEBUG \
            else range(self.epochs)

        use_aphynity = getattr(self.model, "use_aphynity", False)
        logger.debug(f"Train with APHYNITY: {use_aphynity}")
        if use_aphynity:
            lambda_t = self.lambda_0
            for epoch in epochs:
                for _ in range(self.Niter):
                    loss, lpred, norm_Fa = self.compute_loss_aphynity(self.train_data, lambda_t)
                    validation_loss = self.compute_loss(self.test_data)
                    self.losses[epoch] = [loss.detach().cpu().numpy(), validation_loss.detach().cpu().numpy()]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                lambda_t = lambda_t + self.aph_tau * lpred.detach()
                #print(f"\n{epoch}: loss {loss.item()}, lpred {lpred.item()}, |Fa| {norm_Fa.item()}, lambda_t {lambda_t.item()}")
        else:
            for epoch in epochs:
                # Compute loss gradient and step optimizer
                loss = self.compute_loss(self.train_data)
                validation_loss = self.compute_loss(self.test_data)
                self.losses[epoch] = [loss.detach().cpu().numpy(), validation_loss.detach().cpu().numpy()]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def plot_losses(self, filename="loss_{}.png"):
        plt.figure()
        plt.plot(self.losses)
        plt.yscale("log")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(["training", "validation"])
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(self.directory / filename.format(self.model.__class__.__name__))
        plt.grid()
        # plt.show()

    def plot_one_step_predictions(self, data, filename="one_step_pred_{}.png"):
        time, states, actions, next_states = data

        resets = {t+1: states[t+1].unsqueeze(0) for t in np.where(np.diff(time) < 0)[0]}
        pred_states = self.model(states, actions).detach()
        states[[t-1 for t in resets], :] = np.nan
        next_states[[t-1 for t in resets], :] = np.nan
        pred_states[[t-1 for t in resets], :] = np.nan
        actions[[t-1 for t in resets], :] = np.nan

        time = time.cpu().numpy()
        states = states.cpu().numpy()
        pred_states = pred_states.cpu().numpy()
        actions = actions.cpu().numpy()
        fig, axes = plt.subplots(next_states.shape[1] + actions.shape[1], 1, sharex=True)
        for i in range(next_states.shape[1]):
            axes.flat[i].plot(time, next_states[:, i] - pred_states[:, i], 'b', label="error")
            axes.flat[i].set_xlabel("time")
            axes.flat[i].set_ylabel(f"state {i}")
            axes.flat[i].legend(loc="upper right")
        for i in range(actions.shape[1]):
            axes.flat[states.shape[1]+i].plot(time, actions[:, i])
            axes.flat[states.shape[1]+i].set_xlabel("time")
            axes.flat[states.shape[1]+i].set_ylabel(f"action {i}")
            axes.flat[states.shape[1]+i].legend()
        plt.savefig(self.directory / filename.format(self.model.__class__.__name__))

    def plot_integrated_predictions(self, data, filename="integrated_pred_{}.png"):
        time, states, actions, next_states = data

        resets = {t+1: states[t+1].unsqueeze(0) for t in np.where(np.diff(time) < 0)[0]}
        pred_states = self.model.integrate(states[0].unsqueeze(0), actions.unsqueeze(1), resets=resets).squeeze(1)
        states[[t-1 for t in resets], :] = np.nan
        next_states[[t-1 for t in resets], :] = np.nan
        pred_states[[t-1 for t in resets], :] = np.nan
        actions[[t-1 for t in resets], :] = np.nan

        time = time.cpu().numpy()
        next_states = next_states.cpu().numpy()
        states = states.cpu().numpy()
        pred_states = pred_states.cpu().numpy()
        actions = actions.cpu().numpy()
        fig, axes = plt.subplots(states.shape[1] + actions.shape[1], 1, sharex=True)
        for i in range(states.shape[1]):
            axes.flat[i].plot(time, next_states[:, i], 'r', label="ground truth")
            axes.flat[i].plot(time, pred_states[:, i], 'b', label="prediction")
            axes.flat[i].set_xlabel("time")
            axes.flat[i].set_ylabel(f"state {i}")
            axes.flat[i].legend(loc="upper right")
            axes.flat[i].set_ylim([np.nanmin(next_states[:, i]), np.nanmax(next_states[:, i])])
        for i in range(actions.shape[1]):
            axes.flat[states.shape[1]+i].plot(time, actions[:, i])
            axes.flat[states.shape[1]+i].set_xlabel("time")
            axes.flat[states.shape[1]+i].set_ylabel(f"action {i}")
            axes.flat[states.shape[1]+i].legend()
        plt.savefig(self.directory / filename.format(self.model.__class__.__name__))

    def print_model(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)