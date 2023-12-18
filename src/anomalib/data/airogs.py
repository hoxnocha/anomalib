"""Airorgs Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the Airogs dataset.
    The dataset can be found at https://airogs.grand-challenge.org/

References:
    de Vente, C., Vermeer, K. A., Jaccard, N., van Ginneken, B., Lemij, H. G., & Sánchez, C. I. (2021). 
    Rotterdam EyePACS AIROGS train set (1.1.0) [Data set]. IEEE International Symposium on Biomedical Imaging 2022 (ISBI 2022), 
    Kolkata, Calcutta, India. Zenodo. https://doi.org/10.5281/zenodo.5793241

"""

from __future__ import annotations

import logging
from pathlib import Path
from random import sample
from zipfile import ZipFile
import pandas as pd
from typing import Sequence
import glob

import albumentations as A
import os
from pandas import DataFrame
from torch import Tensor, feature_alpha_dropout
from PIL import Image
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.task_type import TaskType
import numpy as np
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    get_transforms,
    
)


logger = logging.getLogger(__name__)

IMG_EXTENSIONS = (".jpg", ".JPG")

DOWNLOAD_INFO = DownloadInfo(
    name="airogs",
    url="https://zenodo.org/records/5793241/files/0.zip?download=1",
    hash="md5:9af51fcaa069c9f0d61dd2cd4e1c05b4",
)

AIROGS_CATEGORIES = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",     
)


import ipdb
def make_airogs_dataset(
    root: str | Path, 
    root_category: str | Path ,
    number_of_samples,
    pre_selection  ,
    split: str | Split | None = None, 
    extensions: Sequence[str] | None = None, 
    
    
) -> DataFrame:
    """Create airogs samples by parsing the airogs data file structure.

    The files are expected to follow the structure:
        path/to/dataset/category/image_filename.jpg
        
    
    This function creates a dataframe to store the parsed information based on the following format:
    |---|-----------------|-------|------------------------------|------------------|-------------|
    |   | path            | split |  image_path                  |       label      | label_index |
    |---|-----------------|-------|------------------------------|------------------|-------------|
    | 0 | path/to/dataset | train | path/to/dataset/filename.jpg |        NRG       |    0        |
    |---|-----------------|-------|------------------------------|------------------|-------------|


    Args:
        root (Path): Path to dataset
        root_category (Path): Path to the category of the dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test). Defaults to None.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional): Boolean to create a validation set from the test set.
            Airogs dataset does not contain a validation set. Those wanting to create a validation set
            could set this flag to ``True``.

    Examples:
        The following example shows how to get training samples from airogs category 0:

        >>> root = Path('/images/innoretvision/eye/airogs')
        >>> category = '0'
        >>> path = root / category
        >>> path
        PosixPath('images/innoretvision/eye/airogs/0')

        >>> samples = make_airogs_dataset(path, split='train',  seed=0, number_of_samples=100)
        >>> samples.head()
             path                               split               image_path                                      label          class_index
        0  images/innoretvision/eye/airogs/0    train        images/innoretvision/eye/airogs/0 TRAIN000000.jpg       NRG             0
        1  images/innoretvision/eye/airogs/0    train        images/innoretvision/eye/airogs/0 TRAIN000001.jpg       NRG             0
        2  images/innoretvision/eye/airogs/0    train        images/innoretvision/eye/airogs/0 TRAIN000002.jpg       NRG             0
        3  images/innoretvision/eye/airogs/0    train        images/innoretvision/eye/airogs/0 TRAIN000003.jpg       NRG             0
        4  images/innoretvision/eye/airogs/0    train        images/innoretvision/eye/airogs/0 TRAIN000004.jpg       NRG             0
    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.
    """
    


     
    
    if extensions is None:
        extensions = IMG_EXTENSIONS
    root = Path(root)
    root_category = Path(root_category)

    csv_file = root / "train_labels.csv"
    if not csv_file.is_file():
        raise FileNotFoundError(f"Could not found {csv_file}")
    

    samples = pd.read_csv(csv_file)
    files =  glob.glob(os.path.join(str(root_category), "*.jpg" ))
    files = [os.path.basename(file)[:-4] for file in files]
    category_files = pd.DataFrame(files, columns=['challenge_id'])
    samples = category_files.merge(samples)
    
    
    samples["challenge_id"] = f"{root_category}" + "/" + samples["challenge_id"] + ".jpg"
    samples = samples.rename(columns={"challenge_id": "image_path", "class":"label"})
    samples = samples[["label","image_path"]]
    
    
    samples.insert(0,"path",f"{root_category}")
    samples.insert(1,"split","train")
    samples.loc[samples.label == "RG" ,"split"] = "test"
    samples.loc[(samples.label == "NRG"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "NRG"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)
    samples["mask_path"] = ""
    
    if pre_selection == True:
        #ipdb.set_trace()
        filted_rows = samples[samples["label"] == "RG"]
        rg_ratio = filted_rows.shape[0] / samples.shape[0]
        filted_rows = filted_rows.sample(n=int(rg_ratio * number_of_samples ), random_state=1)
        select_csv_file = Path("/home/students/tyang/Documents/no_robust_near1003od.csv")
        selected_samples = pd.read_csv(select_csv_file, usecols=[0], names=["image_path"])
        selected_samples.insert(1,"label","NRG")
        #selected_samples = selected_samples[["label","image_path"]]
        selected_samples.insert(0,"path",f"{root_category}")
        selected_samples.insert(1,"split","train")
    
        selected_samples.loc[(selected_samples.label == "NRG"), "label_index"] = LabelName.NORMAL
        selected_samples.loc[(selected_samples.label != "NRG"), "label_index"] = LabelName.ABNORMAL
        selected_samples.label_index = selected_samples.label_index.astype(int)
        selected_samples["mask_path"] = ""
        dfs = [selected_samples, filted_rows]
        samples = pd.concat(dfs)
        
    samples = samples.sample(n=number_of_samples, random_state=1)
    
    
    if split:
        samples = samples[samples.split == split].reset_index(drop=True)
        
    
    
    return samples

class AirogsDataset(AnomalibDataset):
    """Airogs dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
        root (Path | str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. '0'
    """
    
    def __init__(self, root, category: str, number_of_samples, pre_selection , split, task, transform=None,):
        super().__init__(task=task, transform=transform)
        self.root = Path(root)
        self.category = str(category)
        self.root_category = Path(root) / Path(self.category)
        self.number_of_samples = number_of_samples
        #self.data_csv = pd.read_csv(os.path.join(root,'train_labels.csv'))
        
        self.pre_selection = pre_selection
        #self.pre_selection_csv = pre_select_csv
        
        self.split = split
    
    def _setup(self) -> None:
        #ipdb.set_trace()
        self.samples = make_airogs_dataset(
            self.root, 
            self.root_category,
            self.number_of_samples, 
            self.pre_selection,
            #self.pre_selection_csv,
            split=self.split,
            extensions=IMG_EXTENSIONS,
           
            

        )   

class Airogs(AnomalibDataModule):
    """Airogs Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
        category (str): Category of the Airogs dataset (e.g. "0" or "1").
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to None.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to None.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: Path | str,
        category: str,
        number_of_samples,
        pre_selection,
        image_size: int | tuple[int, int] | None = None,
        center_crop: int | tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.CLASSIFICATION,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.2,
        seed: int | None = None,
        
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = str(category)
        self.category = Path(self.category)

        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )

        self.train_data = AirogsDataset(
            root=root,
            category=category,
            number_of_samples=number_of_samples,
            pre_selection=pre_selection,
            task=task,
            transform=transform_train,
            split=Split.TRAIN,
            
        )
        self.test_data = AirogsDataset(
            root=root,
            category=category,
            number_of_samples=number_of_samples,
            pre_selection=pre_selection,
            task=task,
            transform=transform_eval,
            split=Split.TEST,
        )

    #def prepare_data(self) -> None:
        #"""Download the dataset if not available."""
        #if (self.root / self.category ).is_dir():
           # logger.info("Found the dataset.")
        #else:
           # download_and_extract(self.root, DOWNLOAD_INFO)


#if __name__ == "AirogsDataset":
    #import ipdb; ipdb.set_trace()
    #dataset = AirogsDataset(
       # root="/images/innoretvision/eye/airogs", 
        #category="0", 
        #number_of_samples=1300,
       # pre_selection=True,
       # split="split", 
       # task=TaskType.CLASSIFICATION)

