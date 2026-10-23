"""Causal pressure-ensemble inference from arrived pressure measurements."""
from pathlib import Path
import numpy as np
import torch
from torch import nn



def features(history, times, with_time):
    history, times = np.asarray(history), np.asarray(times)
    return np.column_stack((history, times / 60)) if with_time else history.copy()

class ForecastNet(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64),
                                 nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

class Ensemble:
    def __init__(self, directory, seeds, device="cpu"):
        self.device, self.members = device, []
        for seed in seeds:
            state = torch.load(directory / f"seed_{seed}.pth", map_location="cpu", weights_only=True)
            model = ForecastNet(len(state["x_mean"])).to(device)
            model.load_state_dict(state["state_dict"])
            model.eval()
            self.members.append((model, state))

    @torch.no_grad()
    def predict(self, history, times):
        predictions = []
        for model, state in self.members:
            x = features(history, times, state["variant"] == "with_time")
            x = (x-np.asarray(state["x_mean"])) / np.asarray(state["x_scale"])
            y = model(torch.tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy()
            predictions.append(history[:, -1]+y*state["y_scale"]+state["y_mean"])
        values = np.asarray(predictions, dtype=float)
        return values.mean(0), values.std(0, ddof=1)
