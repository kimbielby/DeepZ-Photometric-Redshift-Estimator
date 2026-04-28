"""
config.py

Loads configs/default.yaml and creates a configuration object.
"""
from dataclasses import dataclass
from pathlib import Path
import yaml

# --- Constants ---
MAGNITUDE_COLS = [
    "g_cmodel_mag",
    "r_cmodel_mag",
    "i_cmodel_mag",
    "z_cmodel_mag",
    "y_cmodel_mag",
]

NUM_MAGNITUDE_FEATURES = len(MAGNITUDE_COLS)

IMAGE_EMBEDDING_DIM = 96

SIGMA_NMAD_CONSISTENCY_FACTOR = 1.4826

# Per-channel arcsinh stretch scale factors (noise level per band)
# Computed as std of negative pixels across the full dataset
IMAGE_ARCSINH_SCALE = [0.0212, 0.0352, 0.0369, 0.0688, 0.1367]

# Per-channel mean and std after arcsinh stretch. Used for standardisation
IMAGE_CHANNEL_MEAN = [1.6004, 1.8535, 2.0918, 1.8240, 1.5074]
IMAGE_CHANNEL_STD = [1.8264, 1.8293, 1.9493, 1.8944, 1.7909]

# Credible interval percentiles for PDF summary statistics
# Default: 68% interval (equivalent to ±1σ for a Gaussian)
CI_LOWER_PERCENTILE = 0.16
CI_UPPER_PERCENTILE = 0.84

# --- Dataclasses (1 per YAML section) ---
@dataclass
class PathsConfig:
    """File and directory paths. All resolved to absolute paths at load time."""
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    processed_hdf5_file: Path
    splits_file: Path
    checkpoints_dir: Path
    best_checkpoint: Path
    figures: Path

@dataclass
class DataConfig:
    """Dataset file and split settings"""
    hdf5_path: Path                 # Constructed from raw_dir / hdf5_filename
    chunk_size: int
    val_fraction: float
    test_fraction: float
    random_seed: int
    include_psf: bool

@dataclass
class ModelConfig:
    """Model architecture settings"""
    backbone: str
    use_magnitudes: bool
    magnitude_hidden_dim: int
    num_bins: int
    redshift_min: float
    redshift_max: float

@dataclass
class LRSchedulerConfig:
    enabled: bool
    factor: float
    patience: int
    min_lr: float

@dataclass
class TrainConfig:
    """Training loop and optimiser settings"""
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    loss: str
    use_weighted_sampling: bool
    sampling_weight_power: float
    label_sigma_bins: int
    lr_scheduler: LRSchedulerConfig
    num_workers: int

@dataclass
class EvaluationConfig:
    num_vis_samples: int

@dataclass
class Config:
    """Top-level config object"""
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvaluationConfig

# --- Loader ---
def load_config(path: str | Path) -> Config:
    """
    Load a YAML config file.
    Args:
        path: Path to the YAML configuration file.

    Returns:
        config: Fully populated configuration object with all paths resolved.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    config_path = Path(path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Resolve relative paths against the project root
    project_root = config_path.parent.parent

    d = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # paths
    p = d["paths"]
    paths = PathsConfig(
        project_root=project_root,
        raw_dir=(project_root / p["raw_dir"]).resolve(),
        processed_dir=(project_root / p["processed_dir"]).resolve(),
        processed_hdf5_file=(project_root / p["processed_hdf5_file"]).resolve(),
        splits_file=(project_root / p["splits_file"]).resolve(),
        checkpoints_dir=(project_root / p["checkpoints_dir"]).resolve(),
        best_checkpoint=(project_root / p["best_checkpoint"]).resolve(),
        figures=(project_root / p["figures"]).resolve(),
    )

    # data
    da = d["data"]
    data = DataConfig(
        hdf5_path=(paths.raw_dir / da["hdf5_filename"]).resolve(),
        chunk_size=int(da["chunk_size"]),
        val_fraction=float(da["val_fraction"]),
        test_fraction=float(da["test_fraction"]),
        random_seed=int(da["random_seed"]),
        include_psf=bool(da["include_psf"]),
    )

    # model
    m = d["model"]
    model = ModelConfig(
        backbone=str(m["backbone"]),
        use_magnitudes=bool(m["use_magnitudes"]),
        magnitude_hidden_dim=int(m["magnitude_hidden_dim"]),
        num_bins=int(m["num_bins"]),
        redshift_min=float(m["redshift_min"]),
        redshift_max=float(m["redshift_max"]),
    )

    # training
    t = d["train"]
    lr_sched = t["lr_scheduler"]
    train = TrainConfig(
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        learning_rate=float(t["learning_rate"]),
        patience=int(t["patience"]),
        loss=str(t["loss_function"]),
        use_weighted_sampling=bool(t["use_weighted_sampling"]),
        sampling_weight_power=float(t["sampling_weight_power"]),
        label_sigma_bins=int(t["label_sigma_bins"]),
        lr_scheduler=LRSchedulerConfig(
            enabled=bool(lr_sched["enabled"]),
            factor=float(lr_sched["factor"]),
            patience=int(lr_sched["patience"]),
            min_lr=float(lr_sched["min_lr"]),
        ),
        num_workers=int(t["num_workers"]),

    )

    # evaluation
    e = d["evaluation"]
    evaluation = EvaluationConfig(
        num_vis_samples=int(e["num_vis_samples"]),
    )

    config = Config(
        paths=paths,
        data=data,
        model=model,
        train=train,
        evaluation=evaluation,
    )

    return config





