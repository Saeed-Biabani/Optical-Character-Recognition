from src.schedule import get_linear_schedule_with_warmup
from src.misc import countParams, updateLog, Visualizer
from src.trainutils import TrainOneEpoch, TestOneEpoch
from src.dataset import ImageDataset, loadInfo
from torch.utils.data import DataLoader
from src.tokenizer import loadTokenizer
from src.transforms import Transforms
from src.nn import TRnet, TRconfig
from src.tracker import Tracker
from config import Traincfg
import pandas as pd
import torch

train_cfg = Traincfg()
print(train_cfg)

dataset_info = loadInfo(train_cfg.root)

tokenizer = loadTokenizer(dataset_info.vocab_path)

dataset = ImageDataset(
    train_cfg.root,
    transforms = Transforms(train_cfg.image_size),
    split = 'train'
).train_test_split(test_size = 0.15)

train_ldr = DataLoader(
    dataset['train'],
    train_cfg.batch_size,
    shuffle = True
)

test_ldr = DataLoader(
    dataset['test'],
    train_cfg.batch_size,
    shuffle = True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_cfg = TRconfig(
    image_size = train_cfg.image_size,
    patch_size = train_cfg.patch_size,
    num_channels = train_cfg.num_channels,
    n_vocab = len(tokenizer.vocab),
    max_positions = dataset_info.max_positions,
    pad_idx = dataset_info.pad_idx,
    sos_idx = dataset_info.sos_idx,
    eos_idx = dataset_info.eos_idx
)
print(net_cfg)
model = TRnet(net_cfg).to(device)
print(model)
print(f"Trainable Parameters : {countParams(model):,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = train_cfg.lr,
    weight_decay = train_cfg.weight_decay
)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    train_cfg.n_epochs * 0.05,
    train_cfg.n_epochs
)

log = {
    "epoch" : [],
    "train_loss" : [],
    "test_loss" : [],
    "lr" : [],
}

visualizer = Visualizer(
    patterns = ['loss', 'lr']
)

tracker = Tracker(
    model,
    monitor = train_cfg.monitor_value,
    delta = train_cfg.monitor_delta,
    mode = train_cfg.monitor_mode
)

template = {
    'metadata' : dataset_info.__dict__
}

for epoch in range(1, train_cfg.n_epochs + 1):
    train_log = TrainOneEpoch(
        model = model,
        optimizer = optimizer,
        tokenizer = tokenizer,
        ldr = train_ldr,
        epoch = epoch,
        device = device
    )
    updateLog(train_log, log)

    test_log = TestOneEpoch(
        model = model,
        tokenizer = tokenizer,
        ldr = test_ldr,
        epoch = epoch,
        device = device
    )
    updateLog(test_log, log)

    tracker.step(test_log, epoch)

    log['lr'].append(lr_scheduler.get_last_lr()[0])
    log['epoch'].append(epoch)

    visualizer(log)

    lr_scheduler.step()
    
    tracker.at_epoch_end()
    
    tracker.restore_best_weights(
        template = template,
        fname = train_cfg.save_as
    )
pd.DataFrame(log).to_csv('trainLog.csv', index = False)