from torch.utils.data import DataLoader
from .tokenizer import AutoTokenizer
from typing import Mapping
from .misc import visGrads
import numpy as np
import torch
import tqdm

def TrainOneEpoch(
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    tokenizer : AutoTokenizer,
    ldr : DataLoader,
    epoch : int,
    device : str,
    pro_color : str = 'green',
    desc : str = "[Train] Epoch {} ",
    clip_grad : bool = False
) -> Mapping[str, float]:
    scaler = torch.amp.GradScaler("cuda")
    model.train()

    losses = []

    loop = tqdm.tqdm(
        ldr,
        colour = pro_color,
        desc = desc.format(epoch),
        unit = 'batch'
    )
    for batch_indx, batch in enumerate(loop):
        pixel_values = batch['image'].to(device)
        enc = tokenizer(batch['text'])
        input_ids = enc['input_ids'].to(device)
        input_padding_mask = enc['padding_mask'].to(device)

        labels = torch.cat((input_ids[:, 1:], torch.zeros(input_ids.size(0), 1, device = device).long()), 1)

        with torch.amp.autocast("cuda"):
            __inputs = dict(
                pixel_values = pixel_values,
                input_ids = input_ids,
                input_padding_mask = input_padding_mask,
                labels = labels
            )
            output = model(**__inputs)
            loss = output.loss

        losses.append(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if clip_grad:
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), 1.5
            )
        if batch_indx % 1200 == 0:
            visGrads(model)
        scaler.step(optimizer)
        scaler.update()

        __log = {
            "train_loss" : np.mean(losses)
        }
        loop.set_postfix(__log)
    return __log

@torch.no_grad()
def TestOneEpoch(
    model : torch.nn.Module,
    tokenizer : AutoTokenizer,
    ldr : DataLoader,
    epoch : int,
    device : str,
    pro_color : str = 'yellow',
    desc : str = "[Test] Epoch {} "
) -> Mapping[str, float]:
    model.eval()

    losses = []

    loop = tqdm.tqdm(
        ldr,
        colour = pro_color,
        desc = desc.format(epoch),
        unit = 'batch'
    )
    for batch in loop:

        pixel_values = batch['image'].to(device)

        enc = tokenizer(batch['text'])
        input_ids = enc['input_ids'].to(device)
        input_padding_mask = enc['padding_mask'].to(device)

        labels = torch.cat((input_ids[:, 1:], torch.zeros(input_ids.size(0), 1, device = device).long()), 1)

        with torch.amp.autocast("cuda"):
            __inputs = dict(
                pixel_values = pixel_values,
                input_ids = input_ids,
                input_padding_mask = input_padding_mask,
                labels = labels
            )
            output = model(**__inputs)
            loss = output.loss

        losses.append(loss.item())

        __log = {
            "test_loss" : np.mean(losses)
        }
        loop.set_postfix(__log)
    return __log