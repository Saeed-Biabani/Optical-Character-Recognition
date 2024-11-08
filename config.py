from dataclasses import dataclass

@dataclass
class Traincfg:
    root : str = 'Dataset/Persian-OCR-230k/data/'
    save_as = 'model.pth'

    image_size : tuple[int] = (32, 512)
    patch_size : tuple[int] = (16, 16)
    num_channels : int = 1

    n_epochs : int = 1000

    batch_size : int = 64

    lr : float= 5e-5
    weight_decay : float = 0.01

    monitor_value : str = 'test_loss'
    monitor_mode : str = 'min'
    monitor_delta : int = 0.01