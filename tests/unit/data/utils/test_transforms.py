"""Data transformation test.

This test contains the following test:
    - Transformations could be ``None``, ``yaml``, ``json`` or ``dict``.
    - When it is ``None``, the script loads the default transforms
    - When it is ``yaml``, ``json`` or ``dict``, `albumentations` package
        deserializes the transformations.
"""

import tempfile

import albumentations as A  # noqa: N812
import numpy as np
import pytest
import skimage
import torch

from anomalib.data.utils import get_transforms


def test_transforms_and_image_size_cannot_be_none() -> None:
    """Test when both config and image_size are None.

    When transformations ``config`` and ``image_size`` are ``None``
    ``PreProcessor`` class should raise a ``ValueError``.
    """
    with pytest.raises(ValueError):  # noqa: PT011
        get_transforms(config=None, image_size=None)


@pytest.mark.parametrize(
    ("image_size", "center_crop"),
    [(256, None), ((256, 256), None), (256, 224), (256, (224, 224))],
)
def test_dimensions_can_be_int_or_tuple(image_size: tuple | int, center_crop: tuple | int) -> None:
    """Test combinations of image size and center crop.

    When ``config`` is None, ``image_size`` could be either ``int`` or
    ``Tuple[int, int]``.
    """
    get_transforms(config=None, image_size=image_size, center_crop=center_crop)
    get_transforms(config=None, image_size=image_size, center_crop=center_crop)


@pytest.mark.parametrize(("image_size", "center_crop"), [(256.0, 224), (256, 224.0)])
def test_dimensions_cannot_be_float(image_size: float | int, center_crop: float | int) -> None:
    """Ensure dimensions are not float."""
    with pytest.raises(TypeError):
        get_transforms(config=None, image_size=image_size, center_crop=center_crop)


def test_crop_size_larger_than_image_size() -> None:
    """Ensure crop size is not larger than image size."""
    with pytest.raises(ValueError):  # noqa: PT011
        get_transforms(config=None, image_size=224, center_crop=256)


def test_center_crop_could_be_int_or_tuple() -> None:
    """Ensure center crop is either int or tuple.

    When ``config`` is None, ``image_size`` could be either ``int`` or
    ``Tuple[int, int]``.
    """
    get_transforms(image_size=256)
    get_transforms(image_size=(256, 512))
    with pytest.raises(TypeError):
        get_transforms(config=None, image_size=0.0)


def test_load_transforms_from_string() -> None:
    """Load transforms from yaml.

    When the pre-processor is instantiated via a transform config file, it
    should work with either string or A.Compose and return a ValueError
    otherwise.
    """
    config_path = tempfile.NamedTemporaryFile(suffix=".yaml").name

    # Create a dummy transformation.
    transforms = A.Compose(
        [
            A.Resize(1024, 1024, always_apply=True),
            A.CenterCrop(256, 256, always_apply=True),
            A.Resize(224, 224, always_apply=True),
        ],
    )
    A.save(transform=transforms, filepath_or_buffer=config_path, data_format="yaml")

    # Pass a path to config
    transform = get_transforms(config=config_path)
    assert isinstance(transform, A.Compose)

    # Pass a config of type A.Compose
    transform = get_transforms(config=transforms)
    assert isinstance(transform, A.Compose)

    # Anything else should raise an error
    with pytest.raises(TypeError):
        get_transforms(config=0)


def test_to_tensor_returns_correct_type() -> None:
    """Ensure correct type.

    `to_tensor` flag should ensure that pre-processor returns the expected
    type.
    """
    image = skimage.data.astronaut()

    pre_processor = get_transforms(config=None, image_size=256, to_tensor=True)
    transformed = pre_processor(image=image)["image"]
    assert isinstance(transformed, torch.Tensor)

    pre_processor = get_transforms(config=None, image_size=256, to_tensor=False)
    transformed = pre_processor(image=image)["image"]
    assert isinstance(transformed, np.ndarray)
