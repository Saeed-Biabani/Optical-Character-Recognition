from torchvision.transforms import Compose
from typing import Mapping, Optional
from torch.utils.data import Dataset
from dataclasses import dataclass
from zipfile import ZipFile
from PIL import Image
import pandas as pd
import pathlib
import random
import pickle
import os
import io

class ImageDataset(Dataset):
    def __init__(
        self,
        root, split : str = 'train',
        transforms : Optional[Compose] = None,
        dataset : Optional[pd.DataFrame] = None
    ) -> None:
        super(ImageDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.zip_file = self.__load_zip__(root)
        if dataset is not None:
            self.df = dataset
        else:
            self.df = ImageDataset.load_data(root, split = split)

    @staticmethod
    def load_data(
        root : str,
        split : str= 'train'
    ) -> pd.DataFrame:
        pattern = split if '*' in split else f'{split}*'
        fnames = pathlib.Path(root).glob(pattern)

        dfs = []
        for fname in fnames:
            dfs.append(pd.read_csv(fname))

        return pd.concat(dfs).sample(frac = 1.).reset_index()
    
    def setTransforms(self, transforms):
        self.transforms = transforms
    
    def train_test_split(
        self,
        test_size : int = 0.15
    ) -> Mapping[str, Dataset]:
        n_val_sampels = int(len(self.df) * test_size)
        total_slices = set(range(len(self.df)))

        val_slices = set(random.sample(list(total_slices), n_val_sampels))
        other_ = total_slices - val_slices

        val_data = self.df.iloc[list(val_slices)].sample(frac = 1.)
        train_data = self.df.iloc[list(other_)].sample(frac = 1.)

        return {
            'train' : ImageDataset(
                root = self.root,
                dataset = train_data.reset_index(),
                transforms = self.transforms
            ),
            'test' : ImageDataset(
                root = self.root, 
                dataset = val_data.reset_index(),
                transforms = self.transforms
            )
        }

    def __len__(self) -> None:
        return len(self.df)

    def __load_zip__(self, root : str) -> ZipFile:
        fname = list(pathlib.Path(root).glob('*.zip')).pop()
        return ZipFile(open(fname, 'rb'))

    def __read_image__(self, fname : str) -> Image:
            with self.zip_file.open(fname) as data:
                return Image.open(io.BytesIO(data.read())).convert('L')

    def __getitem__(
        self,
        index : int
    ) -> Mapping[str, object]:
        row = self.df.iloc[index]

        image = self.__read_image__(row['fname'])

        if self.transforms:
            image = self.transforms(image)

        return {
            'image' : image,
            'text' : row['text']
        }

@dataclass
class MetaData:
    vocab_path : str = None

    max_positions : int = None

    sos_idx : int = None
    pad_idx : int = None
    eos_idx : int = None

def loadInfo(root : str) -> MetaData:
    fname = os.path.join(root, 'info.pkl')
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        return MetaData(**data)