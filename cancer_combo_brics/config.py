"""Configuration schema and utilities for CancerCombo."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DataConfig:
    root: str = "./data"
    combination_file: Optional[str] = None
    cell_expression_file: Optional[str] = None
    fg_cache_file: str = "./data/fg_cache.sqlite"
    fg_radius: int = 1
    cell_preprocessor_file: str = "./checkpoints/cell_preprocessor.npz"
    drug_id_col_a: str = "drug_a"
    drug_id_col_b: str = "drug_b"
    smiles_col_a: str = "smiles_a"
    smiles_col_b: str = "smiles_b"
    cell_id_col: str = "cell_line_name"
    dose_col_a: str = "doses_a"
    dose_col_b: str = "doses_b"
    viability_col: str = "viability_matrix"
    split_col: Optional[str] = "split"
    target_scale: str = "percentage_viability"  # percentage_viability: 100% = 100.0
    num_dose_a: int = 4
    num_dose_b: int = 4
    max_fragments: int = 32


@dataclass
class ModelConfig:
    cell_dim: int = 976
    cell_hidden_dim: int = 512
    fragment_dim: int = 512
    mol2vec_native_dim: int = 300
    mol2vec_model_path: str = "data/model_300dim.pkl"
    interaction_hidden_dim: int = 512
    drug_cell_mlp_hidden: int = 512
    num_param_heads: int = 8
    emb_size: int = 1024  # DeepSynBa head embedding dimension matching finalcheck
    param_trunk_hidden: int = 512
    param_dropout: float = 0.2  # DeepSynBa dropout matching finalcheck
    # Pharmacological parameter constraints
    hill_e0: float = 100.0  # Fixed baseline: 100.0 corresponds to 100% viability
    # Dose-dependent bias
    enable_bias: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 50
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 42
    loss_type: str = "huber"  # huber, mse, mae
    huber_delta: float = 0.05


@dataclass
class OptimizerConfig:
    type: str = "AdamW"
    lr_new: float = 1e-4
    # Note: lr_mol2vec removed — pretrained Mol2Vec is frozen (not trainable).
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # cosine, plateau, linear, none
    warmup_epochs: int = 3
    min_lr: float = 1e-6


@dataclass
class EvaluationConfig:
    unseen_drug_split: bool = True
    eval_scenarios: List[int] = field(default_factory=lambda: [1, 2, 3])
    save_surface_plots: bool = True
    max_visualizations: int = 20


@dataclass
class LoggingConfig:
    project_name: str = "CancerCombo"
    experiment_name: str = "functional_group_mol2vec_experiment"
    log_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
