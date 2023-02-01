import os
import cv2
import torch
import numpy as np
import pandas as pd
import pydicom as di
from pandas import DataFrame
from typing import List, Dict
from logging import getLogger
from omegaconf import DictConfig
from joblib import Parallel, delayed

log = getLogger(__name__)


def RSNAresize(dicom_image: np.array, width: int, height: int):
    """
    :param height: image to be resized to
    :param width: image to be resized to
    :param dicom_image: np.array
    :return: dicom_image: np.array
    """
    # Step - I (Resize image to a fixed size)
    # (width, height)
    dicom_image = cv2.resize(dicom_image, (width, height))
    log.info("Resized to dimension 2000x2500")
    return dicom_image


class RSNADatasetModule(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, mode: str = None) -> None:
        super().__init__()
        self.data_dict: Dict = dict()
        self.patient_ids: List = None
        self.data_df: DataFrame = None
        self.cfg = cfg
        self.mode = mode
        self.images_dir = cfg.path.dataset.images.train_full
        if mode == 'full':
            self.data_dir = cfg.path.dataset.csv.train_full
        elif mode == 'train':
            self.data_dir = cfg.path.dataset.csv.train
        elif mode == 'val':
            self.data_dir = cfg.path.dataset.csv.val
        elif mode == 'test':
            self.data_dir = cfg.path.dataset.csv.test
        elif mode == 'submission':
            self.data_dir = cfg.path.dataset.csv.test_submission
            self.images_dir = cfg.path.datase.images.test_submission_images

        self.load_dataframe()
        self.load_images()

    def __len__(self) -> int:
        return len(self.data_df)

    def load_dataframe(self) -> None:
        self.data_df = pd.read_csv(self.data_dir)
        log.warning(f"Loading {self.mode} csv Dataset complete.")

    def process(self, index, filename):
        dicom_image = di.dcmread(filename).pixel_array
        self.data_dict[index] = dicom_image

    def load_images(self) -> None:
        # {0:image_pixel_data1, 1: image_pixel_data2,... }
        image_dir = []
        for i in range(len(self.data_df)):
            patient_id = str(self.data_df.loc[[i]].patient_id[i])
            image_id = str(self.data_df.loc[[i]].image_id[i]) + '.dcm'
            img_dir = os.path.join(self.images_dir, patient_id, image_id)
            image_dir.append(img_dir)

        Parallel(n_jobs=-1)(
            delayed(self.process)(idx, f)
            for idx, f in enumerate(image_dir)
        )
        log.warning(f"Loading {self.mode} image Dataset complete.")

    def __iter__(self):
        return self

    def __getitem__(self, idx: int) -> tuple:
        row = self.data_df.loc[[idx]]
        dicom_image = self.data_dict[idx]
        if self.mode != 'submission':
            label = row['cancer']
        # -- START Transformations --
        dicom_image = RSNAresize(dicom_image, self.cfg.training.resize.width,
                                 self.cfg.training.resize.height)  # Resize image
        # -- END Transformations   --
        dicom_image_tensor = torch.Tensor(dicom_image.astype('int64', casting='same_kind'))  # Convert image to Tensor
        if self.mode != 'submission':
            return dicom_image_tensor, label
        else:
            return dicom_image_tensor
