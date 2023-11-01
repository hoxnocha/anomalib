"""Airorgs Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the Airogs dataset.
    The dataset can be found at https://airogs.grand-challenge.org/

"""

from __future__ import annotations

import logging
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
from typing import Sequence

import albumentations as A
from pandas import DataFrame
from torch import Tensor
from PIL import Image
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    get_transforms,
    extract,
)

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = (".jpg", ".JPG")

DOWNLOAD_INFO = DownloadInfo(
    name="airogs",
    url="/images/innoretvision/eye/airogs"
   
)

CATEGORIES = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",     
)



def make_airogs_dataset(
    root: str | Path, root_category: str | Path, split: str | Split | None = None, extensions: Sequence[str] | None = None
) -> DataFrame:
    """Create airogs samples by parsing the airogs data file structure.

    The files are expected to follow the structure:
        path/to/dataset/category/category/image_filename.jpg
        
    
    This function creates a dataframe to store the parsed information based on the following format:
    |---|---------------|-------|------------------|------------------|-------------|
    |   | path          | split |  image_path      |     label        | label_index |
    |---|---------------|-------|------------------|------------------|-------------|
    | 0 | datasets/name |  test | filename.jpg     |      RG          |     1       |
    |---|---------------|-------|------------------|------------------|-------------|
    | 1 | datasets/name |  test | TRAIN000000.jpg  |     NRG          |     0       |

    Args:
        root (Path): Path to dataset
        root_category (Path): Path to the category of the dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test). Defaults to None.
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional): Boolean to create a validation set from the test set.
            Airogs dataset does not contain a validation set. Those wanting to create a validation set
            could set this flag to ``True``.

    Examples:
        The following example shows how to get training samples from airogs category 0:

        >>> root = Path('/images/innoretvision/eye/airogs')
        >>> category = '0'
        >>> path = root / category / category
        >>> path
        PosixPath('images/innoretvision/eye/airogs/0/0')

        >>> samples = make_airogs_dataset(path, split='train', split_ratio=0, seed=0)
        >>> samples.head()
             path         split            image_path              label          class_index
        0  Airogs/0/0     train        Airogs/0/0/TRAIN000000.jpg   NRG             0
        1  Airogs/0/0     train        Airogs/0/0/TRAIN000175.jpg   NRG             0
        2  Airogs/0/0     train        Airogs/0/0/TRAIN000109.jpg   NRG             0
        3  Airogs/0/0     train        Airogs/0/0/TRAIN000298.jpg   NRG             0
        4  Airogs/0/0     train        Airogs/0/0/TRAIN000309.jpg   NRG             0
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
    
    samples_list = pd.read_csv(csv_file,skiprows=1,header=None)
    samples_list.iloc[0:1] = f"{root_category}" + samples_list.iloc[0:1]
    
    samples = DataFrame(samples_list, columns=["image_path","label"])
    samples.insert(0,"path",f"{root_category}")
    samples.insert(1,"split","train")

    samples.loc[(samples.label == "NRG"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "NRG"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)


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
    
    def __init__(self, root, category: str, task, transform=None):
        super().__init__(task=task, transform=transform)
        self.data_dir = Path(root)
        self.category = category
        self.root_category = Path(root) / Path(category) / Path(category)
        self.data_csv = pd.read_csv(self.data_dir / 'train_labels.csv')

    
    def _setup(self) -> None:
        for zip_file in self.data_dir.glob('*.zip'):
            if zip_file = f"{self.category}.zip":
             extract(zip_file,self.root)
        

        self.samples = make_airogs_dataset(self.data_dir, self.root_category, split=self.split, extensions=IMG_EXTENSIONS)   

             
    
    def __len__(self) -> int:
        return len(self.data_csv)
    
    #def __getitem__(self, index: int) -> dict[str, str | Tensor]:
       # img_id = self.data_csv.iloc[index, 0]
       # label = self.data_csv.iloc[index, 1]
    


class Airogs(AnomalibDataModule):
    """Airogs Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
        category (str): Category of the MVTec dataset (e.g. "bottle" or "cable").
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
        val_split_ratio: float = 0.5,
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
        self.category = Path(category)

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
            task=task, transform=transform_train, split=Split.TRAIN, root=root, category=category
        )
        self.test_data = AirogsDataset(
            task=task, transform=transform_eval, split=Split.TEST, root=root, category=category
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)