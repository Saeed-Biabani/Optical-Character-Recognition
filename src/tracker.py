from typing import Mapping, Optional
from torch import nn
import numpy as np
import torch
import copy

class Tracker:
    def __init__(
        self,
        model : nn.Module,
        monitor : str,
        delta : float,
        mode : str = 'min'
    ) -> None:
        self.model = model
        self.mode = mode
        self.monitor = monitor
        self.delta = delta
        self.best = np.inf if mode == 'min' else -np.inf
        self.best_wights = None
        self.best_epoch = 0
        self.save_if_called = True

    def restore_best_weights(
        self,
        template : Optional[Mapping[str, object]]= {},
        fname : str = 'model.pth'
    ) -> None:
        if self.save_if_called:
            if template is not None:
                if not template.get('model_config', None):
                    template['model_config'] = self.model.cfg.__dict__
                template['model_weights'] = self.best_wights
                torch.save(template, fname)
            else:
                torch.save(self.best_wights, fname)

            self.save_if_called = False
            print(f"[TRACKER] save best weights as {fname}")

    def at_epoch_end(self) -> None:
        print(f"[TRACKER] best {self.monitor} '{self.best}' for epoch '{self.best_epoch}'")
    
    def __need_action__(self, val):
        if self.mode == 'min':
            return (self.best - val) >= self.delta
        elif self.mode == 'max':
            return (self.best - val) <= -self.delta

    def step(
        self,
        log : Mapping[str, float],
        epoch : int
    ) -> None:
        current_val = log.get(self.monitor, np.inf)

        if self.__need_action__(current_val):
            self.best = current_val
            self.best_wights = copy.deepcopy(self.model.state_dict())
            self.best_epoch = epoch
            self.save_if_called = True