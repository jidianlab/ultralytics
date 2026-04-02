# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Label Aliasing System for YOLO datasets.

This module provides functionality to merge multiple dataset label names into single classes,
enabling flexible label mapping through alias configuration.

Usage:
    Configure label aliases in your data.yaml:
    ```yaml
    label_aliases:
      person:  # Target class name
        - human
        - pedestrian
        - people
      vehicle:
        - car
        - truck
        - bus
    ```
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ultralytics.utils import LOGGER


class LabelAliasManager:
    """Manager for label aliasing functionality.

    This class handles the mapping of multiple dataset label names to unified class names,
    enabling seamless merging of labels from different datasets or annotation conventions.

    Attributes:
        aliases (dict[str, list[str]]): Mapping from target class names to source label names.
        alias_to_target (dict[str, str]): Reverse mapping from source labels to target classes.
        target_to_id (dict[str, int]): Mapping from target class names to class IDs.
        id_to_target (dict[int, str]): Mapping from class IDs to target class names.
        original_names (dict[str, int]): Original label names and their indices.

    Methods:
        from_config: Create LabelAliasManager from configuration dictionary.
        map_label_name: Map a label name to its target class name.
        map_label_id: Map an original label ID to target class ID.
        get_unified_names: Get list of unified class names.
        update_labels: Update labels array with alias mappings.
    """

    def __init__(
        self,
        aliases: dict[str, list[str]] | None = None,
        original_names: dict[str, int] | list[str] | None = None,
    ):
        """Initialize LabelAliasManager.

        Args:
            aliases (dict[str, list[str]] | None): Mapping from target class to source labels.
            original_names (dict[str, int] | list[str] | None): Original label names from dataset.
        """
        self.aliases = aliases or {}
        self.alias_to_target: dict[str, str] = {}
        self.target_to_id: dict[str, int] = {}
        self.id_to_target: dict[int, str] = {}
        self.original_to_target_id: dict[int, int] = {}

        # Handle original_names as list or dict
        if isinstance(original_names, list):
            self.original_names = {name: i for i, name in enumerate(original_names)}
        else:
            self.original_names = original_names or {}

        self._build_mappings()

    def _build_mappings(self) -> None:
        """Build internal mapping dictionaries from alias configuration."""
        # Build reverse mapping: source label -> target class
        for target, sources in self.aliases.items():
            # Add target itself as a valid source
            self.alias_to_target[target.lower()] = target
            for source in sources:
                self.alias_to_target[source.lower()] = target

        # Determine unified class names
        unified_names = list(self.aliases.keys()) if self.aliases else []

        # Add any original names that are not aliased
        for name in self.original_names.keys():
            name_lower = name.lower()
            if name_lower not in self.alias_to_target:
                # This name is not aliased, keep it as-is
                self.alias_to_target[name_lower] = name
                if name not in unified_names:
                    unified_names.append(name)

        # Build target name to ID mapping
        self.target_to_id = {name: i for i, name in enumerate(unified_names)}
        self.id_to_target = {i: name for name, i in self.target_to_id.items()}

        # Build original ID to target ID mapping
        for orig_name, orig_id in self.original_names.items():
            target_name = self.alias_to_target.get(orig_name.lower(), orig_name)
            if target_name in self.target_to_id:
                self.original_to_target_id[orig_id] = self.target_to_id[target_name]
            else:
                # Target not found, keep original ID (shouldn't happen normally)
                self.original_to_target_id[orig_id] = orig_id

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LabelAliasManager":
        """Create LabelAliasManager from dataset configuration.

        Args:
            config (dict[str, Any]): Dataset configuration dictionary containing
                'label_aliases' and 'names' keys.

        Returns:
            (LabelAliasManager): Configured alias manager instance.
        """
        aliases = config.get("label_aliases", {})
        names = config.get("names", {})

        # Convert names dict to proper format if needed
        if isinstance(names, dict):
            # names is {id: name} format
            original_names = {v: k for k, v in names.items()}
        elif isinstance(names, list):
            original_names = names
        else:
            original_names = {}

        return cls(aliases=aliases, original_names=original_names)

    def map_label_name(self, name: str) -> str:
        """Map a label name to its target class name.

        Args:
            name (str): Original label name.

        Returns:
            (str): Target class name (or original if no mapping exists).
        """
        return self.alias_to_target.get(name.lower(), name)

    def map_label_id(self, original_id: int) -> int:
        """Map an original label ID to target class ID.

        Args:
            original_id (int): Original class ID from dataset.

        Returns:
            (int): Target class ID after alias mapping.
        """
        return self.original_to_target_id.get(original_id, original_id)

    def get_unified_names(self) -> dict[int, str]:
        """Get dictionary of unified class names.

        Returns:
            (dict[int, str]): Mapping from class ID to class name.
        """
        return self.id_to_target.copy()

    def get_num_classes(self) -> int:
        """Get number of unified classes.

        Returns:
            (int): Number of unique classes after aliasing.
        """
        return len(self.target_to_id)

    def update_labels(self, labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Update labels with alias mappings.

        Args:
            labels (list[dict[str, Any]]): List of label dictionaries with 'cls' arrays.

        Returns:
            (list[dict[str, Any]]): Updated labels with remapped class IDs.
        """
        if not self.original_to_target_id:
            return labels

        # Create mapping array for efficient indexing
        max_orig_id = max(self.original_to_target_id.keys()) + 1
        mapping_array = np.arange(max_orig_id)
        for orig_id, target_id in self.original_to_target_id.items():
            if orig_id < max_orig_id:
                mapping_array[orig_id] = target_id

        for label in labels:
            if "cls" in label and len(label["cls"]) > 0:
                # Map each class ID using array indexing
                cls_array = label["cls"].astype(int).flatten()
                # Handle IDs outside the mapping range
                valid_mask = cls_array < max_orig_id
                mapped_cls = cls_array.copy()
                mapped_cls[valid_mask] = mapping_array[cls_array[valid_mask]]
                label["cls"] = mapped_cls.reshape(label["cls"].shape).astype(label["cls"].dtype)

        return labels

    def update_data_config(self, data: dict[str, Any]) -> dict[str, Any]:
        """Update data configuration with unified class names.

        Args:
            data (dict[str, Any]): Original data configuration.

        Returns:
            (dict[str, Any]): Updated configuration with unified names and nc.
        """
        if not self.aliases:
            return data

        # Update names
        data["names"] = self.get_unified_names()
        data["nc"] = self.get_num_classes()

        # Store alias info for reference
        data["label_alias_manager"] = self

        LOGGER.info(f"Label aliasing: {len(self.original_names)} original classes -> {data['nc']} unified classes")

        return data

    def __repr__(self) -> str:
        """String representation of LabelAliasManager."""
        return (
            f"LabelAliasManager(aliases={len(self.aliases)}, "
            f"original_classes={len(self.original_names)}, "
            f"unified_classes={self.get_num_classes()})"
        )


def apply_label_aliases(
    labels: list[dict[str, Any]],
    alias_manager: LabelAliasManager | None,
) -> list[dict[str, Any]]:
    """Apply label aliasing to a list of labels.

    Args:
        labels (list[dict]): List of label dictionaries.
        alias_manager (LabelAliasManager | None): Alias manager instance.

    Returns:
        (list[dict]): Labels with updated class IDs.
    """
    if alias_manager is None:
        return labels
    return alias_manager.update_labels(labels)


def create_alias_manager_from_data(data: dict[str, Any]) -> LabelAliasManager | None:
    """Create alias manager from data configuration if aliases are defined.

    Args:
        data (dict[str, Any]): Data configuration dictionary.

    Returns:
        (LabelAliasManager | None): Alias manager or None if no aliases configured.
    """
    if "label_aliases" not in data or not data["label_aliases"]:
        return None

    return LabelAliasManager.from_config(data)


def merge_label_aliases(*alias_configs: dict[str, list[str]]) -> dict[str, list[str]]:
    """Merge multiple alias configurations.

    Args:
        *alias_configs: Variable number of alias configuration dictionaries.

    Returns:
        (dict[str, list[str]]): Merged alias configuration.
    """
    merged: dict[str, list[str]] = defaultdict(list)

    for config in alias_configs:
        for target, sources in config.items():
            for source in sources:
                if source not in merged[target]:
                    merged[target].append(source)

    return dict(merged)
