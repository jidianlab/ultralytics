# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .base import BaseDataset
from .build import build_dataloader, build_grounding, build_yolo_dataset, load_inference_source
from .dataset import (
    ClassificationDataset,
    GroundingDataset,
    SemanticDataset,
    YOLOConcatDataset,
    YOLODataset,
    YOLOMultiModalDataset,
)
from .label_alias import LabelAliasManager, apply_label_aliases, create_alias_manager_from_data
from .per_class_augment import (
    ClassAugmentationConfig,
    PerClassAugmentation,
    PerClassAugmentationTransform,
    create_per_class_augmentation,
)

__all__ = (
    "BaseDataset",
    "ClassificationDataset",
    "GroundingDataset",
    "SemanticDataset",
    "YOLOConcatDataset",
    "YOLODataset",
    "YOLOMultiModalDataset",
    "build_dataloader",
    "build_grounding",
    "build_yolo_dataset",
    "load_inference_source",
    # Label aliasing
    "LabelAliasManager",
    "apply_label_aliases",
    "create_alias_manager_from_data",
    # Per-class augmentation
    "ClassAugmentationConfig",
    "PerClassAugmentation",
    "PerClassAugmentationTransform",
    "create_per_class_augmentation",
)
