import cv2
import os
import torch
import numpy as np
import pandas as pd
import pydicom as di
from typing import List
from pandas import DataFrame
from logging import getLogger
from omegaconf import DictConfig

log = getLogger(__name__)


def RSNAresize(dicom_image: np.array):
    """
    :param dicom_image: np.array
    :return: dicom_image: np.array
    """
    # Step - I (Resize image to a fixed size)
    dicom_image = cv2.resize(dicom_image, (2000, 2500))
    log.warning("Resized to dimension 2000x2500")
    return dicom_image


class RSNADatasetModule(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, mode: str = None) -> None:
        super().__init__()
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

    def __len__(self) -> int:
        return len(self.data_df)

    def load_dataframe(self) -> None:
        self.data_df = pd.read_csv(self.data_dir)
        log.info(f"Loading {self.mode} Dataset complete.")

    def __getitem__(self, idx: int) -> tuple:
        row = self.data_df.loc[[idx]]
        patient_id = str(row.patient_id[idx])  # Get Patient id
        image_id = str(row.image_id[idx]) + '.dcm'  # Get Image id
        if self.mode != 'submission':
            label = row.cancer[idx]  # Get Label
        image_dir = os.path.join(self.images_dir, patient_id, image_id)
        dicom_image = di.dcmread(image_dir).pixel_array
        # -- START Transformations --
        dicom_image = RSNAresize(dicom_image)  # Resize image
        # -- END Transformations   --
        dicom_image_tensor = torch.Tensor(dicom_image.astype('int64', casting='same_kind'))  # Convert image to Tensor
        if self.mode != 'submission':
            return dicom_image_tensor, label
        else:
            return dicom_image_tensor
