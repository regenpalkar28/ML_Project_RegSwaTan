"""
Traffic Sign Recognition Dataset and DataLoader Pipeline.
Supports German (GTSRB), Belgian (BelgiumTSC), and Chinese (CTSD) traffic sign benchmarks
with configurable splits, ROI cropping, augmentations, and PyTorch DataLoaders.
"""

import glob
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from config import (
        BATCH_SIZE,
        BELGIAN_DATASET_DIR,
        CHINESE_DATASET_DIR,
        DATASET_NUM_CLASSES,
        DEFAULT_DATASET,
        DEVICE,
        GERMAN_DATASET_DIR,
        IMAGE_SIZE,
        NORM_MEAN,
        NORM_STD,
        NUM_CLASSES,
        NUM_CLASSES_BELGIAN,
        NUM_CLASSES_CHINESE,
        NUM_CLASSES_GERMAN,
        NUM_WORKERS,
        PIN_MEMORY,
        SEED,
        VAL_SPLIT,
    )
except ImportError:
    # Fallback default values
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    GERMAN_DATASET_DIR = DATA_DIR / "German Dataset"
    BELGIAN_DATASET_DIR = DATA_DIR / "Belgian Dataset"
    CHINESE_DATASET_DIR = DATA_DIR / "Chinese Dataset"
    NUM_CLASSES_GERMAN = 43
    NUM_CLASSES_BELGIAN = 62
    NUM_CLASSES_CHINESE = 58
    NUM_CLASSES = 43
    DEFAULT_DATASET = "german"
    DATASET_NUM_CLASSES = {
        "german": 43,
        "gtsrb": 43,
        "belgian": 62,
        "btsc": 62,
        "chinese": 58,
        "ctsd": 58,
    }
    IMAGE_SIZE = (32, 32)
    NORM_MEAN = [0.3403, 0.3121, 0.3214]
    NORM_STD = [0.2724, 0.2608, 0.2669]
    BATCH_SIZE = 64
    VAL_SPLIT = 0.2
    NUM_WORKERS = 0
    PIN_MEMORY = False
    SEED = 42


# ==============================================================================
# GTSRB (German) Class Labels Mapping (0 to 42)
# ==============================================================================
GTSRB_CLASS_NAMES: Dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

CLASS_NAMES = GTSRB_CLASS_NAMES  # Alias for backward compatibility


# ==============================================================================
# Data Transforms
# ==============================================================================
def get_transforms(
    split: str = "train",
    img_size: Tuple[int, int] = IMAGE_SIZE,
    augment: bool = True,
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
) -> transforms.Compose:
    """
    Build torchvision transformation pipeline for traffic sign images.

    Args:
        split (str): 'train', 'val', 'validation', or 'test'.
        img_size (tuple): Target image resolution (Height, Width).
        augment (bool): Whether to apply data augmentation for training.
        norm_mean (list): Normalization channel means.
        norm_std (list): Normalization channel standard deviations.

    Returns:
        transforms.Compose: Composed torchvision transform pipeline.
    """
    mean = norm_mean if norm_mean is not None else NORM_MEAN
    std = norm_std if norm_std is not None else NORM_STD

    is_train = split.lower() == "train"

    if is_train and augment:
        return transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


# ==============================================================================
# 1. German Traffic Sign Recognition Benchmark (GTSRB)
# ==============================================================================
class GTSRBDataset(Dataset):
    """
    PyTorch Dataset for German Traffic Sign Recognition Benchmark (GTSRB, 43 classes).
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = IMAGE_SIZE,
        use_roi_crop: bool = False,
        val_split: float = VAL_SPLIT,
        seed: int = SEED,
        return_meta: bool = False,
    ):
        super().__init__()
        self.root_dir = self._find_dataset_dir(root_dir)
        self.split = split.lower()
        self.img_size = img_size
        self.use_roi_crop = use_roi_crop
        self.val_split = val_split
        self.seed = seed
        self.return_meta = return_meta

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms(
                split=self.split,
                img_size=self.img_size,
                augment=(self.split == "train"),
            )

        self.data_df = self._load_split_df()

    def _find_dataset_dir(self, root_dir: Optional[Union[str, Path]]) -> Path:
        candidates = []
        if root_dir is not None:
            p = Path(root_dir).resolve()
            candidates.extend([
                p,
                p / "German Dataset",
                p / "data" / "German Dataset",
                p / "gtsrb",
                p / "GTSRB",
                p / "data",
            ])

        candidates.extend(
            [
                Path(GERMAN_DATASET_DIR).resolve(),
                Path(__file__).resolve().parent / "German Dataset",
                Path(__file__).resolve().parent.parent / "data" / "German Dataset",
                Path(__file__).resolve().parent,  # data/ directory directly
                Path(__file__).resolve().parent / "GTSRB",
                Path(__file__).resolve().parent / "gtsrb",
                Path("/content/data/German Dataset"),
                Path("/content/German Dataset"),
                Path("/content/GTSRB"),
                Path("/content/data"),
            ]
        )

        for candidate in candidates:
            if candidate.exists() and ((candidate / "Train.csv").exists() or (candidate / "Train").exists()):
                return candidate

        raise FileNotFoundError(
            f"Could not locate GTSRB dataset directory (Train.csv not found).\n"
            f"Checked paths:\n" + "\n".join(f" - {c}" for c in candidates) + "\n"
            f"Please ensure the GTSRB dataset is unzipped and contains 'Train.csv' inside one of these folders, "
            f"or specify --data_dir '<path_to_folder_with_Train.csv>' in your training command."
        )

    def _load_split_df(self) -> pd.DataFrame:
        train_csv_path = self.root_dir / "Train.csv"
        test_csv_path = self.root_dir / "Test.csv"
        meta_csv_path = self.root_dir / "Meta.csv"

        if self.split in ["train", "val", "validation", "all_train"]:
            if not train_csv_path.exists():
                raise FileNotFoundError(f"Missing training annotation file at: {train_csv_path}")
            df = pd.read_csv(train_csv_path)

            if self.split == "all_train" or self.val_split <= 0.0:
                return df.reset_index(drop=True)

            train_df, val_df = train_test_split(
                df,
                test_size=self.val_split,
                stratify=df["ClassId"],
                random_state=self.seed,
            )

            if self.split == "train":
                return train_df.reset_index(drop=True)
            else:
                return val_df.reset_index(drop=True)

        elif self.split == "test":
            if not test_csv_path.exists():
                raise FileNotFoundError(f"Missing test annotation file at: {test_csv_path}")
            return pd.read_csv(test_csv_path).reset_index(drop=True)

        elif self.split == "meta":
            if not meta_csv_path.exists():
                raise FileNotFoundError(f"Missing meta annotation file at: {meta_csv_path}")
            return pd.read_csv(meta_csv_path).reset_index(drop=True)

        else:
            raise ValueError(
                f"Unknown split '{self.split}'. Supported splits: ['train', 'val', 'validation', 'test', 'meta', 'all_train']"
            )

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, dict]]:
        row = self.data_df.iloc[idx]
        rel_path = str(row["Path"]).replace("\\", "/")
        img_path = self.root_dir / rel_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        with Image.open(img_path) as img:
            image = img.convert("RGB")
            if self.use_roi_crop and all(k in row for k in ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]):
                roi_box = (
                    int(row["Roi.X1"]),
                    int(row["Roi.Y1"]),
                    int(row["Roi.X2"]),
                    int(row["Roi.Y2"]),
                )
                image = image.crop(roi_box)

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        label = int(row["ClassId"])

        if self.return_meta:
            meta = {
                "path": str(img_path),
                "rel_path": rel_path,
                "class_id": label,
                "class_name": GTSRB_CLASS_NAMES.get(label, f"Class_{label}"),
                "width": int(row.get("Width", image.width)),
                "height": int(row.get("Height", image.height)),
            }
            if all(k in row for k in ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]):
                meta["roi"] = (
                    int(row["Roi.X1"]),
                    int(row["Roi.Y1"]),
                    int(row["Roi.X2"]),
                    int(row["Roi.Y2"]),
                )
            return image_tensor, label, meta

        return image_tensor, label


# ==============================================================================
# 2. Belgian Traffic Sign Classification Benchmark (BelgiumTSC)
# ==============================================================================
class BelgianDataset(Dataset):
    """
    PyTorch Dataset for Belgium Traffic Sign Classification Benchmark (BelgiumTSC, 62 classes).
    Images are in PPM format stored inside class subfolders (00000 to 00061).
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = IMAGE_SIZE,
        use_roi_crop: bool = False,
        val_split: float = VAL_SPLIT,
        seed: int = SEED,
        return_meta: bool = False,
    ):
        super().__init__()
        self.root_dir = self._find_dataset_dir(root_dir)
        self.split = split.lower()
        self.img_size = img_size
        self.use_roi_crop = use_roi_crop
        self.val_split = val_split
        self.seed = seed
        self.return_meta = return_meta

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms(
                split=self.split,
                img_size=self.img_size,
                augment=(self.split == "train"),
            )

        self.data_df = self._load_split_df()

    def _find_dataset_dir(self, root_dir: Optional[Union[str, Path]]) -> Path:
        candidates = []
        if root_dir is not None:
            p = Path(root_dir).resolve()
            candidates.extend([p, p / "Belgian Dataset", p / "data" / "Belgian Dataset"])

        candidates.extend(
            [
                Path(BELGIAN_DATASET_DIR).resolve(),
                Path(__file__).resolve().parent / "Belgian Dataset",
                Path(__file__).resolve().parent.parent / "data" / "Belgian Dataset",
            ]
        )

        for candidate in candidates:
            if candidate.exists() and (
                (candidate / "BelgiumTSC_Training").exists()
                or (candidate / "Training").exists()
            ):
                return candidate

        raise FileNotFoundError(
            f"Could not locate Belgian Dataset directory. Checked paths: {[str(c) for c in candidates]}"
        )

    def _collect_annotations(self, base_folder: Path) -> pd.DataFrame:
        """Parses all GT-*.csv files inside class subdirectories."""
        records = []
        csv_files = sorted(list(base_folder.glob("**/GT-*.csv")))

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, sep=";")
            except Exception:
                df = pd.read_csv(csv_path, sep=",")

            # Standardize column names (case-insensitive mapping)
            col_map = {}
            for col in df.columns:
                c_clean = str(col).strip()
                c_lower = c_clean.lower()
                if c_lower == "filename":
                    col_map[col] = "Filename"
                elif c_lower == "width":
                    col_map[col] = "Width"
                elif c_lower == "height":
                    col_map[col] = "Height"
                elif c_lower in ["classid", "class_id", "category"]:
                    col_map[col] = "ClassId"
                elif c_lower in ["roi.x1", "roi_x1", "x1"]:
                    col_map[col] = "Roi.X1"
                elif c_lower in ["roi.y1", "roi_y1", "y1"]:
                    col_map[col] = "Roi.Y1"
                elif c_lower in ["roi.x2", "roi_x2", "x2"]:
                    col_map[col] = "Roi.X2"
                elif c_lower in ["roi.y2", "roi_y2", "y2"]:
                    col_map[col] = "Roi.Y2"
                else:
                    col_map[col] = c_clean
            df = df.rename(columns=col_map)

            parent_dir = csv_path.parent
            default_class_id = int(parent_dir.name) if parent_dir.name.isdigit() else 0

            for _, row in df.iterrows():
                fname = str(row.get("Filename", "")).strip()
                img_path = parent_dir / fname
                if img_path.exists():
                    class_id = int(row.get("ClassId", default_class_id))
                    w = int(row.get("Width", 32))
                    h = int(row.get("Height", 32))
                    records.append(
                        {
                            "FullPath": str(img_path),
                            "Filename": fname,
                            "Width": w,
                            "Height": h,
                            "Roi.X1": int(row.get("Roi.X1", 0)),
                            "Roi.Y1": int(row.get("Roi.Y1", 0)),
                            "Roi.X2": int(row.get("Roi.X2", w)),
                            "Roi.Y2": int(row.get("Roi.Y2", h)),
                            "ClassId": class_id,
                        }
                    )

        return pd.DataFrame(records)

    def _load_split_df(self) -> pd.DataFrame:
        # Find Training and Testing base directories
        train_base = self.root_dir / "BelgiumTSC_Training" / "Training"
        if not train_base.exists():
            train_base = self.root_dir / "BelgiumTSC_Training"
        if not train_base.exists():
            train_base = self.root_dir / "Training"

        test_base = self.root_dir / "BelgiumTSC_Testing" / "Testing"
        if not test_base.exists():
            test_base = self.root_dir / "BelgiumTSC_Testing"
        if not test_base.exists():
            test_base = self.root_dir / "Testing"

        if self.split in ["train", "val", "validation", "all_train"]:
            if not train_base.exists():
                raise FileNotFoundError(f"Missing Belgian training directory at: {train_base}")

            df = self._collect_annotations(train_base)
            if len(df) == 0:
                raise ValueError(f"No valid annotations found in Belgian training directory: {train_base}")

            if self.split == "all_train" or self.val_split <= 0.0:
                return df.reset_index(drop=True)

            train_df, val_df = train_test_split(
                df,
                test_size=self.val_split,
                stratify=df["ClassId"],
                random_state=self.seed,
            )

            if self.split == "train":
                return train_df.reset_index(drop=True)
            else:
                return val_df.reset_index(drop=True)

        elif self.split == "test":
            if not test_base.exists():
                raise FileNotFoundError(f"Missing Belgian testing directory at: {test_base}")

            df = self._collect_annotations(test_base)
            if len(df) == 0:
                raise ValueError(f"No valid annotations found in Belgian testing directory: {test_base}")
            return df.reset_index(drop=True)

        else:
            raise ValueError(f"Unknown split '{self.split}' for BelgianDataset.")

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, dict]]:
        row = self.data_df.iloc[idx]
        img_path = Path(row["FullPath"])

        with Image.open(img_path) as img:
            image = img.convert("RGB")
            if self.use_roi_crop:
                roi_box = (
                    int(row["Roi.X1"]),
                    int(row["Roi.Y1"]),
                    int(row["Roi.X2"]),
                    int(row["Roi.Y2"]),
                )
                image = image.crop(roi_box)

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        label = int(row["ClassId"])

        if self.return_meta:
            meta = {
                "path": str(img_path),
                "filename": str(row["Filename"]),
                "class_id": label,
                "class_name": f"Belgian_Class_{label:02d}",
                "width": int(row["Width"]),
                "height": int(row["Height"]),
                "roi": (
                    int(row["Roi.X1"]),
                    int(row["Roi.Y1"]),
                    int(row["Roi.X2"]),
                    int(row["Roi.Y2"]),
                ),
            }
            return image_tensor, label, meta

        return image_tensor, label


# ==============================================================================
# 3. Chinese Traffic Sign Dataset (CTSD)
# ==============================================================================
class ChineseDataset(Dataset):
    """
    PyTorch Dataset for Chinese Traffic Sign Dataset (CTSD, 58 classes).
    Annotations are stored in annotations.csv with corresponding images in images/ folder.
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = IMAGE_SIZE,
        use_roi_crop: bool = False,
        val_split: float = VAL_SPLIT,
        test_split: float = 0.15,
        seed: int = SEED,
        return_meta: bool = False,
    ):
        super().__init__()
        self.root_dir = self._find_dataset_dir(root_dir)
        self.split = split.lower()
        self.img_size = img_size
        self.use_roi_crop = use_roi_crop
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.return_meta = return_meta

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms(
                split=self.split,
                img_size=self.img_size,
                augment=(self.split == "train"),
            )

        self.data_df = self._load_split_df()

    def _find_dataset_dir(self, root_dir: Optional[Union[str, Path]]) -> Path:
        candidates = []
        if root_dir is not None:
            p = Path(root_dir).resolve()
            candidates.extend([p, p / "Chinese Dataset", p / "data" / "Chinese Dataset"])

        candidates.extend(
            [
                Path(CHINESE_DATASET_DIR).resolve(),
                Path(__file__).resolve().parent / "Chinese Dataset",
                Path(__file__).resolve().parent.parent / "data" / "Chinese Dataset",
            ]
        )

        for candidate in candidates:
            if candidate.exists() and (
                (candidate / "annotations.csv").exists() or (candidate / "images").exists()
            ):
                return candidate

        raise FileNotFoundError(
            f"Could not locate Chinese Dataset directory. Checked paths: {[str(c) for c in candidates]}"
        )

    def _load_split_df(self) -> pd.DataFrame:
        csv_path = self.root_dir / "annotations.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing Chinese dataset annotations file at: {csv_path}")

        df = pd.read_csv(csv_path)

        # Standardize column mappings: file_name, width, height, x1, y1, x2, y2, category
        df = df.rename(
            columns={
                "category": "ClassId",
                "x1": "Roi.X1",
                "y1": "Roi.Y1",
                "x2": "Roi.X2",
                "y2": "Roi.Y2",
            }
        )

        if self.split == "all":
            return df.reset_index(drop=True)

        # Stratified train/val/test split
        # First separate test split
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_split,
            stratify=df["ClassId"],
            random_state=self.seed,
        )

        if self.split == "test":
            return test_df.reset_index(drop=True)

        # Calculate relative val split proportion
        rel_val_split = self.val_split / (1.0 - self.test_split)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=rel_val_split,
            stratify=train_val_df["ClassId"],
            random_state=self.seed,
        )

        if self.split == "train":
            return train_df.reset_index(drop=True)
        elif self.split in ["val", "validation"]:
            return val_df.reset_index(drop=True)
        else:
            raise ValueError(f"Unknown split '{self.split}' for ChineseDataset.")

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, dict]]:
        row = self.data_df.iloc[idx]
        file_name = str(row["file_name"])
        img_path = self.root_dir / "images" / file_name
        if not img_path.exists():
            img_path = self.root_dir / file_name

        if not img_path.exists():
            raise FileNotFoundError(f"Chinese dataset image not found: {img_path}")

        with Image.open(img_path) as img:
            image = img.convert("RGB")
            if self.use_roi_crop and all(k in row for k in ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]):
                roi_box = (
                    int(row["Roi.X1"]),
                    int(row["Roi.Y1"]),
                    int(row["Roi.X2"]),
                    int(row["Roi.Y2"]),
                )
                image = image.crop(roi_box)

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        label = int(row["ClassId"])

        if self.return_meta:
            meta = {
                "path": str(img_path),
                "file_name": file_name,
                "class_id": label,
                "class_name": f"Chinese_Class_{label:02d}",
                "width": int(row.get("width", image.width)),
                "height": int(row.get("height", image.height)),
                "roi": (
                    int(row.get("Roi.X1", 0)),
                    int(row.get("Roi.Y1", 0)),
                    int(row.get("Roi.X2", image.width)),
                    int(row.get("Roi.Y2", image.height)),
                ),
            }
            return image_tensor, label, meta

        return image_tensor, label


# ==============================================================================
# Unified Dataset Wrapper & Factory
# ==============================================================================
def get_dataset(
    dataset_name: str = "german",
    split: str = "train",
    root_dir: Optional[Union[str, Path]] = None,
    transform: Optional[Callable] = None,
    img_size: Tuple[int, int] = IMAGE_SIZE,
    use_roi_crop: bool = False,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
    return_meta: bool = False,
) -> Dataset:
    """
    Unified dataset instantiation factory for any supported traffic sign benchmark.

    Args:
        dataset_name (str): 'german' ('gtsrb'), 'belgian' ('btsc'), or 'chinese' ('ctsd').
        split (str): 'train', 'val' / 'validation', or 'test'.
        root_dir (str or Path, optional): Custom path to dataset.
        transform (Callable, optional): Custom transform.
        img_size (tuple): Image dimensions (Height, Width).
        use_roi_crop (bool): Whether to crop image to ROI bounding box.
        val_split (float): Validation split ratio.
        seed (int): Random seed for reproducibility.
        return_meta (bool): Whether to return metadata alongside image and label.

    Returns:
        Dataset: Instantiated PyTorch Dataset instance.
    """
    name = dataset_name.lower().strip()

    if name in ["german", "gtsrb"]:
        return GTSRBDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            img_size=img_size,
            use_roi_crop=use_roi_crop,
            val_split=val_split,
            seed=seed,
            return_meta=return_meta,
        )
    elif name in ["belgian", "btsc", "belgium"]:
        return BelgianDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            img_size=img_size,
            use_roi_crop=use_roi_crop,
            val_split=val_split,
            seed=seed,
            return_meta=return_meta,
        )
    elif name in ["chinese", "ctsd", "china"]:
        return ChineseDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            img_size=img_size,
            use_roi_crop=use_roi_crop,
            val_split=val_split,
            seed=seed,
            return_meta=return_meta,
        )
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Available: ['german', 'belgian', 'chinese']"
        )


class TrafficSignDataset(Dataset):
    """
    Unified wrapper class for any traffic sign benchmark dataset.
    """

    def __init__(
        self,
        dataset_name: str = "german",
        split: str = "train",
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = IMAGE_SIZE,
        use_roi_crop: bool = False,
        val_split: float = VAL_SPLIT,
        seed: int = SEED,
        return_meta: bool = False,
    ):
        self.underlying_dataset = get_dataset(
            dataset_name=dataset_name,
            split=split,
            root_dir=root_dir,
            transform=transform,
            img_size=img_size,
            use_roi_crop=use_roi_crop,
            val_split=val_split,
            seed=seed,
            return_meta=return_meta,
        )

    def __len__(self) -> int:
        return len(self.underlying_dataset)

    def __getitem__(self, idx: int):
        return self.underlying_dataset[idx]


# ==============================================================================
# Unified DataLoader Pipeline Factory
# ==============================================================================
def get_dataloaders(
    dataset_name: str = "german",
    root_dir: Optional[Union[str, Path]] = None,
    batch_size: int = BATCH_SIZE,
    img_size: Tuple[int, int] = IMAGE_SIZE,
    val_split: float = VAL_SPLIT,
    num_workers: int = NUM_WORKERS,
    use_roi_crop: bool = False,
    pin_memory: bool = PIN_MEMORY,
    seed: int = SEED,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns standard PyTorch DataLoaders for train, validation, and test splits
    for German, Belgian, or Chinese traffic sign datasets.

    Args:
        dataset_name (str): 'german', 'belgian', or 'chinese'.
        root_dir (str or Path, optional): Custom dataset directory path.
        batch_size (int): Batch size for DataLoader.
        img_size (tuple): Target image resolution (Height, Width).
        val_split (float): Ratio of training data to use for validation.
        num_workers (int): Worker subprocess count.
        use_roi_crop (bool): Whether to crop images to ROI bounding box.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        seed (int): Random seed for reproducible splitting.
        train_transform (Callable, optional): Custom transform for training.
        eval_transform (Callable, optional): Custom transform for val and test.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    train_dataset = get_dataset(
        dataset_name=dataset_name,
        split="train",
        root_dir=root_dir,
        transform=train_transform,
        img_size=img_size,
        use_roi_crop=use_roi_crop,
        val_split=val_split,
        seed=seed,
    )

    val_dataset = get_dataset(
        dataset_name=dataset_name,
        split="val",
        root_dir=root_dir,
        transform=eval_transform,
        img_size=img_size,
        use_roi_crop=use_roi_crop,
        val_split=val_split,
        seed=seed,
    )

    test_dataset = get_dataset(
        dataset_name=dataset_name,
        split="test",
        root_dir=root_dir,
        transform=eval_transform,
        img_size=img_size,
        use_roi_crop=use_roi_crop,
        val_split=val_split,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# ==============================================================================
# Self-Test and Validation Block
# ==============================================================================
if __name__ == "__main__":
    from models.RawCNN import get_model

    datasets_to_test = [
        ("German (GTSRB)", "german", NUM_CLASSES_GERMAN),
        ("Belgian (BelgiumTSC)", "belgian", NUM_CLASSES_BELGIAN),
        ("Chinese (CTSD)", "chinese", NUM_CLASSES_CHINESE),
    ]

    print("=" * 65)
    print("Multi-Dataset Pipeline Test: German, Belgian, and Chinese")
    print("=" * 65)

    for ds_label, ds_key, num_classes in datasets_to_test:
        print(f"\n>>> Testing {ds_label} Pipeline (Target Classes: {num_classes})")
        train_l, val_l, test_l = get_dataloaders(
            dataset_name=ds_key,
            batch_size=32,
            img_size=(32, 32),
            use_roi_crop=False,
            num_workers=0,
        )

        n_train = len(train_l.dataset)
        n_val = len(val_l.dataset)
        n_test = len(test_l.dataset)
        print(f"  Train samples:      {n_train:,}")
        print(f"  Validation samples: {n_val:,}")
        print(f"  Test samples:       {n_test:,}")
        print(f"  Total samples:      {n_train + n_val + n_test:,}")

        # Fetch sample batch
        images, labels = next(iter(train_l))
        print(f"  Batch image shape:  {images.shape} (Expected: [32, 3, 32, 32])")
        print(f"  Batch label shape:  {labels.shape} (Expected: [32])")
        print(f"  Label range:        min={labels.min().item()}, max={labels.max().item()} (max < {num_classes})")

        # Test model forward pass
        model = get_model(num_classes=num_classes)
        logits = model(images)
        print(f"  Model output shape: {logits.shape} (Expected: [32, {num_classes}])")
        assert logits.shape == (32, num_classes), f"Output shape mismatch for {ds_key}"
        print(f"  => {ds_label} PASS!")

    print("\n" + "=" * 65)
    print("All 3 datasets successfully verified!")
    print("=" * 65)
