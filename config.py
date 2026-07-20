import os
try:
    import yaml
except ImportError:
    yaml = None
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    molformer_in_dim: int
    morgan_in_dim: int
    descriptor_in_dim: int
    cell_in_dim: int
    use_pathway_projection: bool
    n_pathways: int
    molformer_model_name: str
    use_pretrained_molformer: bool
    enable_drug_drug_attention: bool
    use_symmetric_fusion: bool
    e_min: float
    e_max: float
    c_min: float
    c_max: float
    h_min: float
    h_max: float
    alpha_min: float
    alpha_max: float

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    device: str
    checkpoint_dir: str
    save_top_k: int
    num_workers: int
    seed: int
    optimizer_name: str = "AdamW"
    scheduler_name: str = "ReduceLROnPlateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3

def load_config(config_path: str = "config.yaml") -> tuple[ModelConfig, TrainingConfig]:
    """Loads configuration parameters from config.yaml and returns dataclass objects.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        tuple[ModelConfig, TrainingConfig]: Configuration dataclasses.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        if yaml is not None:
            config_dict = yaml.safe_load(f)
        else:
            content = f.read()
            config_dict = {}
            curr = None
            for line in content.splitlines():
                line_clean = line.split('#')[0].rstrip()
                if not line_clean.strip():
                    continue
                if not line.startswith(' ') and line_clean.endswith(':'):
                    curr = line_clean.strip()[:-1]
                    config_dict[curr] = {}
                elif curr and ':' in line_clean:
                    k, v = line_clean.split(':', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if v.lower() == 'true': v = True
                    elif v.lower() == 'false': v = False
                    else:
                        try:
                            if '.' in v or 'e' in v.lower(): v = float(v)
                            else: v = int(v)
                        except ValueError:
                            pass
                    config_dict[curr][k] = v
        
    model_data = config_dict["model"]
    training_data = config_dict["training"]
    
    # Ensure types are correct from YAML loading
    model_config = ModelConfig(
        d_model=int(model_data["d_model"]),
        n_heads=int(model_data["n_heads"]),
        d_ff=int(model_data["d_ff"]),
        dropout=float(model_data["dropout"]),
        molformer_in_dim=int(model_data["molformer_in_dim"]),
        morgan_in_dim=int(model_data["morgan_in_dim"]),
        descriptor_in_dim=int(model_data["descriptor_in_dim"]),
        cell_in_dim=int(model_data["cell_in_dim"]),
        use_pathway_projection=bool(model_data["use_pathway_projection"]),
        n_pathways=int(model_data["n_pathways"]),
        molformer_model_name=str(model_data["molformer_model_name"]),
        use_pretrained_molformer=bool(model_data["use_pretrained_molformer"]),
        enable_drug_drug_attention=bool(model_data["enable_drug_drug_attention"]),
        use_symmetric_fusion=bool(model_data["use_symmetric_fusion"]),
        e_min=float(model_data["e_min"]),
        e_max=float(model_data["e_max"]),
        c_min=float(model_data["c_min"]),
        c_max=float(model_data["c_max"]),
        h_min=float(model_data["h_min"]),
        h_max=float(model_data["h_max"]),
        alpha_min=float(model_data["alpha_min"]),
        alpha_max=float(model_data["alpha_max"])
    )
    
    training_config = TrainingConfig(
        batch_size=int(training_data["batch_size"]),
        epochs=int(training_data["epochs"]),
        lr=float(training_data["lr"]),
        weight_decay=float(training_data["weight_decay"]),
        device=str(training_data["device"]),
        checkpoint_dir=str(training_data["checkpoint_dir"]),
        save_top_k=int(training_data["save_top_k"]),
        num_workers=int(training_data["num_workers"]),
        seed=int(training_data["seed"]),
        optimizer_name=str(training_data.get("optimizer_name", "AdamW")),
        scheduler_name=str(training_data.get("scheduler_name", "ReduceLROnPlateau")),
        scheduler_factor=float(training_data.get("scheduler_factor", 0.5)),
        scheduler_patience=int(training_data.get("scheduler_patience", 3))
    )
    
    return model_config, training_config
