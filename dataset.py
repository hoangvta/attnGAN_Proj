from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
import pandas as pd
import torch
import numpy as np

from torch.utils.data import DataLoader

import PIL
import os
import re

import config.settings as config

data_path = 'flowers/'
df = pd.read_pickle(data_path + 'captions.pickle')

encoded_text_list = df[0]
text_dict = df[2]
rev_text_dict = df[3]


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.base_folder = Path(self.root) / "flowers"
        self.image_folder = self.base_folder / "jpg"
        self.caption_folder = self.base_folder / "txt"

        self._number_of_image = config.IMAGE_NUMBER


        self._loadData()

    def _loadData(self):
        from scipy.io import loadmat

        labels = loadmat(self.base_folder / "imagelabels.mat", squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._images = []
        self._captions = []
        _image_files = []
        _caption_files = []

        for id in range(1, self._number_of_image):
            _image_files.append(self.image_folder / f"image_{id:05d}.jpg")
            _caption_files.append(self.caption_folder / f"image_{id:05d}.txt")
            self._labels.append(image_id_to_label[id] - 1)

        for image_file in _image_files:
            self._images.append(PIL.Image.open(image_file).convert('RGB'))

        for caption_file in _caption_files:
            with open(caption_file, 'rb') as f:
                caption = filter(None, f.read().decode('utf8').split("\n"))
                caption = [re.sub('[^a-zA-Z0-9]+', ' ', item) for item in caption]
                caption = [list(map(lambda word: rev_text_dict[word], item.split())) for item in caption]
            self._captions.append(caption)

        del _image_files
        del _caption_files
        del image_id_to_label
        del labels

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        image = self._images[idx]

        dice = np.random.randint(0, 10)

        captions = self._captions[idx][dice]
        caption_lengths = len(captions)
        captions = np.pad(captions, (0, config.SENTENCE_SIZE - caption_lengths))
        label = self._labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            captions = self.target_transform(captions)
            caption_lengths = self.target_transform(caption_lengths)

        return image, captions, caption_lengths, label
