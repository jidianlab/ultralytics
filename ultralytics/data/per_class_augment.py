# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Per-Class Data Augmentation for YOLO datasets.

This module provides functionality for applying different augmentation parameters
to different object classes based on their pixel regions.

Usage:
    Configure per-class augmentations in your data.yaml:
    ```yaml
    class_augmentations:
      person:  # Class name (or alias)
        hsv_h: 0.02
        hsv_s: 0.8
        hsv_v: 0.5
        region_expand: 0.1  # Expand region by 10% (< 1 = ratio)
      vehicle:
        hsv_h: 0.01
        hsv_s: 0.5
        hsv_v: 0.3
        region_expand: 10  # Expand region by 10 pixels (>= 1 = pixels)
    ```
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import cv2
import numpy as np

from ultralytics.utils import LOGGER


class ClassAugmentationConfig:
    """Configuration for class-specific augmentation parameters.

    Attributes:
        class_name (str): Name of the class this config applies to.
        hsv_h (float): HSV hue augmentation fraction.
        hsv_s (float): HSV saturation augmentation fraction.
        hsv_v (float): HSV value (brightness) augmentation fraction.
        degrees (float): Rotation degrees.
        translate (float): Translation fraction.
        scale (float): Scale gain.
        shear (float): Shear degrees.
        perspective (float): Perspective fraction.
        flipud (float): Vertical flip probability.
        fliplr (float): Horizontal flip probability.
        region_expand (float): Region expansion (pixels if >= 1, ratio if < 1).
        brightness (float): Brightness adjustment factor.
        contrast (float): Contrast adjustment factor.
        blur (float): Blur kernel size (0 = no blur).
        noise (float): Gaussian noise standard deviation.
    """

    def __init__(
        self,
        class_name: str,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        degrees: float = 0.0,
        translate: float = 0.0,
        scale: float = 0.0,
        shear: float = 0.0,
        perspective: float = 0.0,
        flipud: float = 0.0,
        fliplr: float = 0.0,
        region_expand: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        blur: float = 0.0,
        noise: float = 0.0,
    ):
        """Initialize ClassAugmentationConfig.

        Args:
            class_name (str): Name of the class.
            hsv_h (float): HSV hue fraction (default: 0.015).
            hsv_s (float): HSV saturation fraction (default: 0.7).
            hsv_v (float): HSV value fraction (default: 0.4).
            degrees (float): Rotation degrees (default: 0.0).
            translate (float): Translation fraction (default: 0.0).
            scale (float): Scale gain (default: 0.0).
            shear (float): Shear degrees (default: 0.0).
            perspective (float): Perspective fraction (default: 0.0).
            flipud (float): Vertical flip probability (default: 0.0).
            fliplr (float): Horizontal flip probability (default: 0.0).
            region_expand (float): Region expansion amount (default: 0.0).
            brightness (float): Brightness adjustment (default: 0.0).
            contrast (float): Contrast adjustment (default: 0.0).
            blur (float): Blur kernel size (default: 0.0).
            noise (float): Gaussian noise std (default: 0.0).
        """
        self.class_name = class_name
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flipud = flipud
        self.fliplr = fliplr
        self.region_expand = region_expand
        self.brightness = brightness
        self.contrast = contrast
        self.blur = blur
        self.noise = noise

    @classmethod
    def from_dict(cls, class_name: str, config: dict[str, Any]) -> "ClassAugmentationConfig":
        """Create ClassAugmentationConfig from dictionary.

        Args:
            class_name (str): Name of the class.
            config (dict[str, Any]): Configuration dictionary.

        Returns:
            (ClassAugmentationConfig): Configured instance.
        """
        return cls(
            class_name=class_name,
            hsv_h=config.get("hsv_h", 0.015),
            hsv_s=config.get("hsv_s", 0.7),
            hsv_v=config.get("hsv_v", 0.4),
            degrees=config.get("degrees", 0.0),
            translate=config.get("translate", 0.0),
            scale=config.get("scale", 0.0),
            shear=config.get("shear", 0.0),
            perspective=config.get("perspective", 0.0),
            flipud=config.get("flipud", 0.0),
            fliplr=config.get("fliplr", 0.0),
            region_expand=config.get("region_expand", 0.0),
            brightness=config.get("brightness", 0.0),
            contrast=config.get("contrast", 0.0),
            blur=config.get("blur", 0.0),
            noise=config.get("noise", 0.0),
        )

    def get_augmentation_strength(self) -> float:
        """Calculate total augmentation strength coefficient.

        Used to determine priority when regions overlap.
        Lower values indicate less aggressive augmentation.

        Returns:
            (float): Augmentation strength coefficient.
        """
        return (
            abs(self.hsv_h) + abs(self.hsv_s) + abs(self.hsv_v)
            + abs(self.degrees) / 180.0 + abs(self.translate)
            + abs(self.scale) + abs(self.shear) / 90.0
            + abs(self.brightness) + abs(self.contrast)
            + abs(self.blur) / 10.0 + abs(self.noise) / 50.0
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"ClassAugmentationConfig(class={self.class_name}, strength={self.get_augmentation_strength():.3f})"


class PerClassAugmentation:
    """Per-class data augmentation handler.

    Applies different augmentation parameters to different object classes
    based on their pixel regions in the image.

    Attributes:
        class_configs (dict[str, ClassAugmentationConfig]): Per-class configurations.
        default_config (ClassAugmentationConfig): Default augmentation config.
        class_name_to_id (dict[str, int]): Mapping from class names to IDs.

    Methods:
        apply: Apply per-class augmentations to image and labels.
        get_class_config: Get augmentation config for a class.
        compute_region_mask: Compute augmentation regions with expansion.
    """

    def __init__(
        self,
        class_configs: dict[str, ClassAugmentationConfig] | None = None,
        default_config: ClassAugmentationConfig | None = None,
        class_name_to_id: dict[str, int] | None = None,
    ):
        """Initialize PerClassAugmentation.

        Args:
            class_configs (dict[str, ClassAugmentationConfig] | None): Per-class configs.
            default_config (ClassAugmentationConfig | None): Default config.
            class_name_to_id (dict[str, int] | None): Class name to ID mapping.
        """
        self.class_configs = class_configs or {}
        self.default_config = default_config or ClassAugmentationConfig("default")
        self.class_name_to_id = class_name_to_id or {}
        self.id_to_class_name = {v: k for k, v in self.class_name_to_id.items()}

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        class_names: dict[int, str] | list[str] | None = None,
    ) -> "PerClassAugmentation":
        """Create PerClassAugmentation from configuration.

        Args:
            config (dict[str, Any]): Configuration with 'class_augmentations' key.
            class_names (dict[int, str] | list[str] | None): Class name mapping.

        Returns:
            (PerClassAugmentation): Configured instance.
        """
        class_aug_config = config.get("class_augmentations", {})

        # Build class configs
        class_configs = {}
        for class_name, aug_params in class_aug_config.items():
            class_configs[class_name.lower()] = ClassAugmentationConfig.from_dict(
                class_name, aug_params
            )

        # Build name to ID mapping
        if isinstance(class_names, dict):
            class_name_to_id = {v.lower(): k for k, v in class_names.items()}
        elif isinstance(class_names, list):
            class_name_to_id = {name.lower(): i for i, name in enumerate(class_names)}
        else:
            class_name_to_id = {}

        # Create default config from global augmentation params
        default_config = ClassAugmentationConfig(
            class_name="default",
            hsv_h=config.get("hsv_h", 0.015),
            hsv_s=config.get("hsv_s", 0.7),
            hsv_v=config.get("hsv_v", 0.4),
            degrees=config.get("degrees", 0.0),
            translate=config.get("translate", 0.1),
            scale=config.get("scale", 0.5),
            shear=config.get("shear", 0.0),
            perspective=config.get("perspective", 0.0),
            flipud=config.get("flipud", 0.0),
            fliplr=config.get("fliplr", 0.5),
        )

        return cls(class_configs, default_config, class_name_to_id)

    def get_class_config(self, class_id: int) -> ClassAugmentationConfig:
        """Get augmentation config for a class ID.

        Args:
            class_id (int): Class ID.

        Returns:
            (ClassAugmentationConfig): Configuration for the class.
        """
        class_name = self.id_to_class_name.get(class_id, "").lower()
        return self.class_configs.get(class_name, self.default_config)

    def has_custom_augmentation(self, class_id: int) -> bool:
        """Check if a class has custom augmentation config.

        Args:
            class_id (int): Class ID.

        Returns:
            (bool): True if custom config exists.
        """
        class_name = self.id_to_class_name.get(class_id, "").lower()
        return class_name in self.class_configs

    def compute_expanded_bbox(
        self,
        bbox: np.ndarray,
        expand: float,
        img_shape: tuple[int, int],
        other_bboxes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute expanded bounding box with boundary and overlap constraints.

        Args:
            bbox (np.ndarray): Original bbox [x1, y1, x2, y2].
            expand (float): Expansion amount (pixels if >= 1, ratio if < 1).
            img_shape (tuple[int, int]): Image shape (height, width).
            other_bboxes (np.ndarray | None): Other bboxes to avoid overlapping.

        Returns:
            (np.ndarray): Expanded bbox [x1, y1, x2, y2].
        """
        h, w = img_shape
        x1, y1, x2, y2 = bbox

        # Calculate expansion amount
        if expand < 1.0:
            # Ratio-based expansion
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            expand_x = bbox_w * expand
            expand_y = bbox_h * expand
        else:
            # Pixel-based expansion
            expand_x = expand_y = expand

        # Expand bbox
        new_x1 = max(0, x1 - expand_x)
        new_y1 = max(0, y1 - expand_y)
        new_x2 = min(w, x2 + expand_x)
        new_y2 = min(h, y2 + expand_y)

        # Check for overlap with other bboxes and constrain
        if other_bboxes is not None and len(other_bboxes) > 0:
            for other in other_bboxes:
                ox1, oy1, ox2, oy2 = other

                # Skip if it's the same bbox
                if np.allclose([x1, y1, x2, y2], other):
                    continue

                # Check for overlap and constrain expansion
                # Left side expansion constraint
                if new_x1 < ox2 and x1 >= ox2:
                    new_x1 = max(new_x1, ox2)

                # Right side expansion constraint
                if new_x2 > ox1 and x2 <= ox1:
                    new_x2 = min(new_x2, ox1)

                # Top side expansion constraint
                if new_y1 < oy2 and y1 >= oy2:
                    new_y1 = max(new_y1, oy2)

                # Bottom side expansion constraint
                if new_y2 > oy1 and y2 <= oy1:
                    new_y2 = min(new_y2, oy1)

        return np.array([new_x1, new_y1, new_x2, new_y2])

    def create_augmentation_mask(
        self,
        img_shape: tuple[int, int],
        bboxes: np.ndarray,
        classes: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, list[np.ndarray]]]:
        """Create augmentation mask assigning pixels to class-specific augmentations.

        When regions overlap, uses the augmentation with smaller coefficient.

        Args:
            img_shape (tuple[int, int]): Image shape (height, width).
            bboxes (np.ndarray): Bounding boxes [N, 4] in xyxy format.
            classes (np.ndarray): Class IDs [N].

        Returns:
            (tuple[np.ndarray, dict]): Augmentation mask and class-to-regions mapping.
        """
        h, w = img_shape

        # Mask: -1 = no specific augmentation, >=0 = class-specific augmentation index
        aug_mask = np.full((h, w), -1, dtype=np.int32)

        # Track which regions belong to which classes
        class_regions: dict[int, list[np.ndarray]] = {}

        # Sort bboxes by augmentation strength (lowest first for priority)
        bbox_list = []
        for i, (bbox, cls_id) in enumerate(zip(bboxes, classes)):
            cls_id = int(cls_id)
            config = self.get_class_config(cls_id)
            strength = config.get_augmentation_strength()
            bbox_list.append((i, bbox, cls_id, config, strength))

        # Sort by augmentation strength (smaller = higher priority)
        bbox_list.sort(key=lambda x: x[4])

        # Process bboxes in order of augmentation strength
        for i, bbox, cls_id, config, strength in bbox_list:
            # Skip if no custom augmentation
            if not self.has_custom_augmentation(cls_id):
                continue

            # Compute expanded bbox
            expanded_bbox = self.compute_expanded_bbox(
                bbox, config.region_expand, img_shape, bboxes
            )

            x1, y1, x2, y2 = expanded_bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Only update pixels that haven't been assigned yet
            # (maintains priority for smaller augmentation coefficients)
            region_mask = aug_mask[y1:y2, x1:x2] == -1
            aug_mask[y1:y2, x1:x2][region_mask] = cls_id

            # Track region
            if cls_id not in class_regions:
                class_regions[cls_id] = []
            class_regions[cls_id].append(expanded_bbox)

        return aug_mask, class_regions

    def apply_hsv_augmentation(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        config: ClassAugmentationConfig,
        class_id: int,
    ) -> np.ndarray:
        """Apply HSV augmentation to specific region.

        Args:
            img (np.ndarray): Input image (BGR).
            mask (np.ndarray): Boolean mask for region.
            config (ClassAugmentationConfig): Augmentation config.
            class_id (int): Class ID for logging.

        Returns:
            (np.ndarray): Augmented image.
        """
        if not np.any(mask):
            return img

        img = img.copy()

        # Convert region to HSV
        region = img[mask]
        if len(region) == 0:
            return img

        # Reshape for HSV conversion
        region_2d = region.reshape(-1, 1, 3)
        hsv_region = cv2.cvtColor(region_2d, cv2.COLOR_BGR2HSV)

        # Apply random HSV augmentation
        h_gain = 1 + np.random.uniform(-config.hsv_h, config.hsv_h)
        s_gain = 1 + np.random.uniform(-config.hsv_s, config.hsv_s)
        v_gain = 1 + np.random.uniform(-config.hsv_v, config.hsv_v)

        hsv_region = hsv_region.astype(np.float32)
        hsv_region[..., 0] = (hsv_region[..., 0] * h_gain) % 180
        hsv_region[..., 1] = np.clip(hsv_region[..., 1] * s_gain, 0, 255)
        hsv_region[..., 2] = np.clip(hsv_region[..., 2] * v_gain, 0, 255)

        # Convert back to BGR
        hsv_region = hsv_region.astype(np.uint8)
        bgr_region = cv2.cvtColor(hsv_region, cv2.COLOR_HSV2BGR)

        # Apply to image
        img[mask] = bgr_region.reshape(-1, 3)

        return img

    def apply_brightness_contrast(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        config: ClassAugmentationConfig,
    ) -> np.ndarray:
        """Apply brightness and contrast augmentation to region.

        Args:
            img (np.ndarray): Input image.
            mask (np.ndarray): Boolean mask for region.
            config (ClassAugmentationConfig): Augmentation config.

        Returns:
            (np.ndarray): Augmented image.
        """
        if config.brightness == 0 and config.contrast == 0:
            return img

        if not np.any(mask):
            return img

        img = img.copy()

        # Apply brightness and contrast
        alpha = 1.0 + np.random.uniform(-config.contrast, config.contrast)
        beta = np.random.uniform(-config.brightness, config.brightness) * 255

        img[mask] = np.clip(img[mask].astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return img

    def apply_blur(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        config: ClassAugmentationConfig,
    ) -> np.ndarray:
        """Apply blur augmentation to region.

        Args:
            img (np.ndarray): Input image.
            mask (np.ndarray): Boolean mask for region.
            config (ClassAugmentationConfig): Augmentation config.

        Returns:
            (np.ndarray): Augmented image.
        """
        if config.blur <= 0:
            return img

        if not np.any(mask):
            return img

        img = img.copy()

        # Create blurred version
        ksize = int(config.blur)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # Apply to region
        img[mask] = blurred[mask]

        return img

    def apply_noise(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        config: ClassAugmentationConfig,
    ) -> np.ndarray:
        """Apply Gaussian noise to region.

        Args:
            img (np.ndarray): Input image.
            mask (np.ndarray): Boolean mask for region.
            config (ClassAugmentationConfig): Augmentation config.

        Returns:
            (np.ndarray): Augmented image.
        """
        if config.noise <= 0:
            return img

        if not np.any(mask):
            return img

        img = img.copy()

        # Generate noise
        noise = np.random.normal(0, config.noise, img.shape).astype(np.float32)
        noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Apply to region
        img[mask] = noisy_img[mask]

        return img

    def apply(
        self,
        img: np.ndarray,
        bboxes: np.ndarray,
        classes: np.ndarray,
    ) -> np.ndarray:
        """Apply per-class augmentations to image.

        Args:
            img (np.ndarray): Input image (BGR, HWC format).
            bboxes (np.ndarray): Bounding boxes [N, 4] in xyxy pixel coordinates.
            classes (np.ndarray): Class IDs [N].

        Returns:
            (np.ndarray): Augmented image.
        """
        if len(bboxes) == 0 or not self.class_configs:
            return img

        h, w = img.shape[:2]

        # Create augmentation mask
        aug_mask, class_regions = self.create_augmentation_mask((h, w), bboxes, classes)

        # Apply augmentations for each class
        for cls_id, regions in class_regions.items():
            config = self.get_class_config(cls_id)

            # Create boolean mask for this class
            class_mask = aug_mask == cls_id

            if not np.any(class_mask):
                continue

            # Apply HSV augmentation
            img = self.apply_hsv_augmentation(img, class_mask, config, cls_id)

            # Apply brightness/contrast
            img = self.apply_brightness_contrast(img, class_mask, config)

            # Apply blur
            img = self.apply_blur(img, class_mask, config)

            # Apply noise
            img = self.apply_noise(img, class_mask, config)

        return img

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PerClassAugmentation(classes={len(self.class_configs)}, "
            f"total_classes={len(self.class_name_to_id)})"
        )


class PerClassAugmentationTransform:
    """Transform wrapper for per-class augmentation.

    This class can be used in the augmentation pipeline to apply
    per-class augmentations during data loading.

    Attributes:
        augmenter (PerClassAugmentation): Per-class augmentation handler.
        p (float): Probability of applying augmentation.

    Methods:
        __call__: Apply transform to labels dictionary.
    """

    def __init__(
        self,
        config: dict[str, Any],
        class_names: dict[int, str] | list[str] | None = None,
        p: float = 1.0,
    ):
        """Initialize PerClassAugmentationTransform.

        Args:
            config (dict[str, Any]): Configuration dictionary.
            class_names (dict[int, str] | list[str] | None): Class names.
            p (float): Probability of applying augmentation.
        """
        self.augmenter = PerClassAugmentation.from_config(config, class_names)
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Apply per-class augmentation to labels.

        Args:
            labels (dict[str, Any]): Labels dictionary with 'img', 'instances'.

        Returns:
            (dict[str, Any]): Augmented labels.
        """
        if np.random.random() > self.p:
            return labels

        img = labels.get("img")
        if img is None:
            return labels

        instances = labels.get("instances")
        if instances is None or len(instances) == 0:
            return labels

        # Get bboxes and classes
        bboxes = instances.bboxes  # Should be in xyxy format
        classes = instances.cls

        if len(bboxes) == 0:
            return labels

        # Convert normalized bboxes to pixel coordinates if needed
        h, w = img.shape[:2]
        if bboxes.max() <= 1.0:  # Normalized
            bboxes_pixel = bboxes.copy()
            bboxes_pixel[:, [0, 2]] *= w
            bboxes_pixel[:, [1, 3]] *= h
        else:
            bboxes_pixel = bboxes

        # Apply augmentation
        labels["img"] = self.augmenter.apply(img, bboxes_pixel, classes)

        return labels

    def __repr__(self) -> str:
        """String representation."""
        return f"PerClassAugmentationTransform(p={self.p}, {self.augmenter})"


def create_per_class_augmentation(
    data_config: dict[str, Any],
    class_names: dict[int, str] | list[str] | None = None,
) -> PerClassAugmentation | None:
    """Create per-class augmentation from data configuration.

    Args:
        data_config (dict[str, Any]): Data configuration dictionary.
        class_names (dict[int, str] | list[str] | None): Class name mapping.

    Returns:
        (PerClassAugmentation | None): Augmenter or None if not configured.
    """
    if "class_augmentations" not in data_config or not data_config["class_augmentations"]:
        return None

    return PerClassAugmentation.from_config(data_config, class_names)
