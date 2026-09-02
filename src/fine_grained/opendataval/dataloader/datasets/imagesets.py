"""TorchVision data sets.

Uses `torchvision <https://github.com/pytorch/vision>`_. as a dependency.
"""

import os
from pathlib import Path
from typing import TypeVar, Union

import matplotlib as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    MNIST,
    STL10,
    SVHN,
    FashionMNIST,
    VisionDataset,
)

from opendataval.dataloader.register import Register
from opendataval.dataloader.util import FolderDataset

Self = TypeVar("Self", bound=Dataset)


class CIFAR10DataLoader:
    """Handles CIFAR-10 data loading with the standard augmentation pipeline."""

    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)

    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir

    def get_dataloader(self, batch_size: int = 128, split: str = 'train',
                       augment: bool = False, num_workers: int = 4,
                       pin_memory: bool = True):
        print(f"Loading CIFAR-10 augment={augment}")
        if augment:
            # Standard CIFAR-10 augmentation: pad-and-crop + horizontal flip.
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])

        is_train = (split == 'train')
        dataset = CIFAR10(
            root=self.data_dir, download=True, train=is_train, transform=transform,
        )
        return DataLoader(
            dataset, shuffle=is_train, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        )


def ResnetEmbeding(
    dataset_class: type[VisionDataset],
    size: tuple[int, int] = (224, 224),
    batch_size: int = 128,
):
    """Convert PIL color Images into embeddings with ResNet50 model.

    Given a PIL Images, passes through ResNet50 (as done by prior Data Valuation papers)
    and saves the vector embeddings. The embeddings are extracted from the ``avgpool``
    layer of ResNet50. The extraction is through the PyTorch forward hook feature.

    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun,
        Deep Residual Learning for Image Recognition,
        2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
        Jun. 2016, doi: https://doi.org/10.1109/cvpr.2016.90.
    .. [2] A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    Parameters
    ----------
    image_set : type[VisionDataset]
        Class of Dataset to compute the embeddings of.
    size : tuple[int, int], optional
        Size to resize images to, by default (224, 224)

    Returns
    -------
    Callable
        Wrapped function when called returns a covariate embedding array and label array
    """

    def wrapper(
        cache_dir: str, force_download: bool, *args, **kwargs
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Methods: `@christiansafka <https://github.com/christiansafka/img2vec>`_."""
        from torchvision.models.resnet import ResNet50_Weights, resnet50

        img2vec_transforms = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                # Means and std as specified by @christiansafka
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        cache_dir = Path(cache_dir)
        embed_path = cache_dir / f"{dataset_class.__name__}_embed/"

        # Resnet inputs expect `img2vec_transforms`ed images as input
        data = dataset_class(
            root=cache_dir,
            download=force_download or not cache_dir.exists(),
            transform=img2vec_transforms,
            *args,
            **kwargs,
        )

        if FolderDataset.exists(embed_path):
            return FolderDataset.load(embed_path), data.targets

        # Slow down on gpu vs cpu is quite substantial, uses gpu accel if available
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Gets the avgpool layer, the outputs of this layer are our embeddings
        embedder = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        embedder.fc = nn.Identity()
        folder_dataset = FolderDataset(embed_path)

        # We will register a hook to extract the ouput of avgpool layers.
        labels_list = []

        with torch.no_grad():  # Passes through model, and our hook extracts outputs
            for batch_num, (img, labels) in tqdm.tqdm(
                enumerate(DataLoader(data, batch_size, pin_memory=True, num_workers=4))
            ):
                img = img.to(device)
                embedding = embedder(img).detach().cpu()
                labels_list.extend(labels)

                folder_dataset.write(batch_num, embedding)

        folder_dataset.save()
        return folder_dataset, np.array(labels_list)

    return wrapper


def show_image(imgs: Union[list[Image.Image], Image.Image]) -> None:
    """Displays an image or a list of images."""
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return


class VisionAdapter(Dataset):
    """Adapter for PyTorch vision data sets. __call__ is called by :py:class:`Register`.

    Adapter for MNIST data sets. __init__ inputs the class and __call__ initializes the
    Dataset and extracts labels. __call__ returns tuple[Self, np.array] where Self is
    a Dataset of covariates and np.array is an array of labels.

    Parameters
    ----------
    dataset_class : type[VisionDataset]
        Torchvision data set class provided.
    """

    def __init__(self, dataset_class: type[VisionDataset]):
        self.dataset_class = dataset_class
        self.transform = None  # Additional transforms applied to the wrapper Dataset.

    def __call__(
        self, cache_dir: str, force_download: bool, *args, **kwargs
    ) -> tuple[Self, np.ndarray]:
        """Return covariates as PyTorch Dataset and labels as np.array.

        Parameters
        ----------
        cache_dir : str
            Directory to download cached files to.
        force_download : bool
            Whether to force a download of the data files.

        Returns
        -------
        tuple[Self, np.ndarray]
            Returns covariates as PyTorch Dataset and labels as np.array. This approach
            was chosen because we need to perform vectorized operations on the labels
            in some data valuators but not necessarily on the covariates, thus, to save
            memory, we leave the Covariates as a PyTorch Dataset.
        """
        # force_download is set to true if  directory doesn't exist, initial download
        force_download = force_download or not os.path.exists(cache_dir)
        self.dataset = self.dataset_class(
            root=cache_dir, download=force_download, *args, **kwargs
        )
        labels = np.array(self.dataset.targets, dtype=int)

        # Incase we forget to apply transform, ensures output is tensor
        if self.dataset.transform is None:
            self.transform = transforms.ToTensor()

        return self, labels

    def __getitem__(self, index: int) -> torch.Tensor:
        """Getitem extracts only the covariates.

        Parameters
        ----------
        index : int
            Index to get covariate from the dataset

        Returns
        -------
        torch.Tensor
            Tensor representing the image with transforms added
        """
        img, _ = self.dataset.__getitem__(index)  # Ignores label
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.dataset)


numbers = Register("mnist", True, True)(VisionAdapter(MNIST))
"""Vision Classification data set registered as ``"mnist"``, from TorchVision."""

fashion = Register("fashion", True, True)(VisionAdapter(FashionMNIST))
"""Vision Classification data set registered as ``"fashion"``, from TorchVision."""

cifar100 = Register("cifar100", True, True)(VisionAdapter(CIFAR100))
"""Vision Classification data set registered as ``"cifar100"``, from TorchVision."""

# CIFAR10 presplit loader - loads data as deterministic, normalized tensors.
# NOTE: No RandomCrop/RandomHorizontalFlip here. Baking a *single* random crop/flip
# into a stack()'d tensor at load time means every epoch trains on the exact same
# augmented pixels forever (no augmentation diversity across epochs). Per-epoch
# augmentation is instead applied on-the-fly inside ResNet110.fit() via
# `_CIFAR10AugmentedTrainDataset`, which re-samples crop/flip fresh every __getitem__ call.
def _load_cifar10_presplit(cache_dir, force_download=False, train_count=None, valid_count=None, test_count=None, **kwargs):
    # Both train and test pools get the SAME deterministic transform: normalization only.
    # This tensor is the canonical, unaugmented source of truth consumed by every
    # DataEvaluator (Shapley/OOB/KNN/LAVA/etc.) as well as by ResNet110.fit(),
    # which applies its own fresh augmentation.
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Load CIFAR-10 (no augmentation baked in at this stage)
    tr = CIFAR10(root=cache_dir, train=True, transform=base_transform, download=True)
    te = CIFAR10(root=cache_dir, train=False, transform=base_transform, download=True)

    # Use specified counts, or all data if not specified
    n_train = train_count if train_count is not None else len(tr)
    n_test = test_count if test_count is not None else len(te)

    # Stack into tensors, respecting specified counts
    xt = torch.stack([tr[i][0] for i in range(n_train)])
    yt = np.array(tr.targets[:n_train], dtype=int)

    xte = torch.stack([te[i][0] for i in range(n_test)])
    yte = np.array(te.targets[:n_test], dtype=int)

    # Return: (train_pool=n_train normalized, valid_placeholder=empty, test=n_test normalized)
    # valid_count is ignored for presplit mode (validation split from train pool happens in fetcher)
    return xt, xt[:0], xte, yt, yt[:0], yte

cifar10 = Register("cifar10", True, True)
cifar10.presplit = True
cifar10.covar_label_func = _load_cifar10_presplit


def _load_cifar10_resnet9_presplit(cache_dir, force_download=False, model_path="./Embed_model/model.pth", **kwargs):
    """Load CIFAR-10 with ResNet9 embeddings, presplit (train + test).

    Follows the same pattern as _load_cifar10_presplit:
    - Loads train (50K) + test (10K) = 60K total
    - Applies ResNet9 embedding model
    - Caches embeddings to avoid regeneration
    - Returns presplit format for split_dataset_by_count
    """
    from opendataval.model.resnet import ResNet9

    cache_dir = Path(cache_dir)
    embed_cache_path = cache_dir / "cifar10_resnet9_embeddings.pt"

    # Check if embeddings are cached
    if embed_cache_path.exists():
        print(f"[ResNet9] Loading cached embeddings from {embed_cache_path}...")
        cached_data = torch.load(embed_cache_path, weights_only=False)
        return cached_data['xt'], cached_data['xt_empty'], cached_data['xte'], cached_data['yt'], cached_data['yt_empty'], cached_data['yte']

    # Preprocessing transforms for ResNet9
    img_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

    # Load both train and test from CIFAR10
    tr = CIFAR10(root=cache_dir, train=True, transform=img_transforms, download=True)
    te = CIFAR10(root=cache_dir, train=False, transform=img_transforms, download=True)

    # Load ResNet9 model
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    embedder = ResNet9(num_classes=10)
    embedder.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Remove classification layers for embeddings
    embedder.model[9] = nn.Identity()
    embedder.model[10] = nn.Identity()
    embedder = embedder.to(device)
    embedder.eval()

    # Generate embeddings for train
    print("[ResNet9] Generating train embeddings...")
    train_embeddings = []
    with torch.no_grad():
        for img, _ in tqdm.tqdm(tr, total=len(tr)):
            img = img.unsqueeze(0).to(device)
            embedding = embedder(img).detach().cpu()
            train_embeddings.append(embedding.squeeze(0))

    xt = torch.stack(train_embeddings)
    yt = np.array(tr.targets, dtype=int)

    # Generate embeddings for test
    print("[ResNet9] Generating test embeddings...")
    test_embeddings = []
    with torch.no_grad():
        for img, _ in tqdm.tqdm(te, total=len(te)):
            img = img.unsqueeze(0).to(device)
            embedding = embedder(img).detach().cpu()
            test_embeddings.append(embedding.squeeze(0))

    xte = torch.stack(test_embeddings)
    yte = np.array(te.targets, dtype=int)

    # Cache embeddings for future use
    print(f"[ResNet9] Caching embeddings to {embed_cache_path}...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'xt': xt,
        'xt_empty': xt[:0],
        'xte': xte,
        'yt': yt,
        'yt_empty': yt[:0],
        'yte': yte
    }, embed_cache_path)
    print("[ResNet9] ✓ Embeddings cached")

    # Return presplit format: (train_pool, valid_placeholder, test_pool)
    return xt, xt[:0], xte, yt, yt[:0], yte


cifar10_resnet9_embed = Register("cifar10-embedding-resnet9", True, True)
cifar10_resnet9_embed.presplit = True
cifar10_resnet9_embed.covar_label_func = _load_cifar10_resnet9_presplit
"""Vision Classification registered as ``"cifar10"``, from TorchVision."""

cifar10_embed = Register("cifar10-embeddings", True, True)(ResnetEmbeding(CIFAR10))
"""Vision Classification registered as ``"cifar10-embeddings"`` ResNet50 embeddings"""

stl10_embed = Register("stl10-embeddings", True, True)(ResnetEmbeding(STL10))
"""Vision Classification registered as ``"stl10-embeddings"`` ResNet50 embeddings"""

svhn_embed = Register("svhn-embeddings", True, True)(ResnetEmbeding(SVHN))
"""Vision Classification registered as ``"svhn-embeddings"`` ResNet50 embeddings"""
