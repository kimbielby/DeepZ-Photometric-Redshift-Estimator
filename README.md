# DeepZ: Photometric Redshift Estimation from HSC Galaxy Images 

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green) 

DeepZ is a deep learning framework for photometric redshift estimation from five-band Hyper Suprime-Cam (HSC) galaxy images. It uses a custom Inception-based convolutional neural network trained on the GalaxiesML-Spectra dataset - 113,245 galaxies with HSC grizy imaging and DESI spectroscopic redshifts - to produce full redshift probability density functions (PDFs) over the range z = 0 to 4. The model is trained using Continuous Ranked Probability Score (CRPS) loss, which respects the ordinal structure of redshift bins and naturally produces well-calibrated PDFs suitable for downstream cosmological analyses.  

## Table of Contents 

- [Features](#features)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Models](#models)
- [Citation](#citation)
- [License](#license) 
- [References](#references) 
 
## Features

- Custom Inception-based CNN designed from scratch for 5-band astronomical flux images, following Pasquet et al. (2019) and Treyer et al. (2024) 
- Classification over 200 redshift bins (z = 0-4) producing a full PDF per galaxy rather than a point estimate 
- CRPS loss function respecting the ordinal structure of redshift bins, with support for soft Gaussian labels to improve PDF calibration 
- Optional HSC grizy photometric magnitude branch fused with image features 
- Redshift-weighted sampling to address class imbalance in spectroscopic training sets, with configurable weighting power 
- Evaluation with both point estimate metrics (sigma_NMAD, bias, outlier rate) and PDF quality metrics (CRPS, PIT) 
- Resume training from checkpoint with optional history appending 
- Configurable via a single YAML file - no hardcoded parameters 
- Full preprocessing pipeline included, handling arcsinh stretch and per-channel standardisation of HSC flux images 
- Jupyter notebooks for preprocessing, training, evaluation and inference
- Command line pipelines for training and evaluation 

## Results

DeepZ was evaluated across four configurations to study the effect of redshift-weighted sampling on point estimate accuracy and PDF calibration. All models share the same architecture, loss function and hyperparameters - only the sampling strategy and label type differ. 

### Quantitative Results 

| Model | Sampling power | σ_NMAD | Bias | Outlier rate | CRPS | PIT calibration |
|-------|---------------|--------|------|--------------|------|-----------------|
| Baseline | 0.0 | 0.0126 | 0.0002 | 0.0070 | 0.9070 | Poor |
| Model 1 | 1.0 | 0.0163 | 0.0070 | 0.0089 | 1.3920 | Good |
| Model 2 | 0.5 | 0.0151 | 0.0080 | 0.0064 | 1.1288 | Good |
| Model 3 | 0.3 | **0.0144** | 0.0073 | 0.0067 | **1.0877** | Good | 

Metrics are computed on the held-out test set of 16,986 galaxies. 

- **σ_NMAD** — normalised median absolute deviation of (z_pred - z_true) / (1 + z_true)
- **Bias** — median of (z_pred - z_true) / (1 + z_true)
- **Outlier rate** — fraction of galaxies with |Δz / (1 + z_true)| > 0.15
- **CRPS** — mean Continuous Ranked Probability Score over the test set
- **PIT calibration** — qualitative assessment based on PIT histogram shape 

### Key Findings 

- The baseline achieves the best point estimate accuracy (σ_NMAD = 0.0126) and near-zero bias, but produces systematically overconfident PDFs evidenced by a strong spike at PIT = 1.0 across all redshift ranges 
- Aggressive reweighting (Model 1, power = 1.0) dramatically improves PDF calibration but at the cost of point estimate accuracy 
- **Model 3 (power = 0.3) achieves the best balance** - recovering most of the baseline's point estimate accuracy whilst eliminating the systematic PDF bias present in the unweighted baseline 
- Temperature scaling was investigated as a post-training calibration method but was found to be ineffective - the PDF miscalibration is a training-time problem requiring weighted sampling to address 

### Training Curves 

Training curves, PIT histograms, scatter plots and mean PDF visualisations for all models are available in `outputs/figures/`. 

## Project Structure

```
deepz-photometric-redshift-estimator/
├── config/
│   ├── baseline_config.yaml      # Archived baseline configuration
│   ├── model1_config.yaml        # Archived Model 1 configuration
│   ├── model2_config.yaml        # Archived Model 2 configuration
│   ├── model3_config.yaml        # Archived Model 3 configuration
│   └── default.yaml              # Active configuration file
├── data/
│   ├── processed/                # Preprocessed HDF5 file (generated)
│   ├── raw/                      # Raw HDF5 file (not tracked by git)
│   └── splits.csv                # Train/val/test splits (generated)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_train_evaluate.ipynb
│   ├── 04_inference.ipynb
│   └── 05_demo.ipynb
├── outputs/
│   ├── checkpoints/              # Model checkpoints and training history (generated)
│   ├── figures/                  # Generated plots and visualisations (generated)
│   ├── baseline_results.json     # Baseline evaluation metrics (generated)
│   ├── model1_results.json       # Model 1 evaluation metrics (generated)
│   ├── model2_results.json       # Model 2 evaluation metrics (generated)
│   └── model3_results.json       # Model 3 evaluation metrics (generated)
├── pzest/
│   ├── dataset/
│   │   ├── dataset.py            # GalaxyDataset and DataLoader
│   │   └── splits.py             # Train/val/test split generation
│   ├── evaluation/
│   │   ├── calibrate.py          # Temperature scaling
│   │   ├── inference.py          # PDF prediction
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── test.py               # Test set evaluation
│   ├── models/
│   │   ├── deepz.py              # Full DeepZ model
│   │   ├── inception.py          # Inception backbone
│   │   └── magnitude.py          # Magnitude MLP branch
│   ├── pipelines/
│   │   ├── evaluate_pipeline.py  # Evaluation pipeline
│   │   └── train_pipeline.py     # Training pipeline
│   ├── training/
│   │   ├── loss.py               # CRPS loss function
│   │   ├── sampling.py           # Weighted sampling utilities
│   │   ├── trainer.py            # Training loop
│   │   └── validation.py         # Validation loop
│   ├── config.py                 # Configuration dataclasses and loader
│   ├── preprocessing.py          # Image preprocessing pipeline
│   └── utils.py                  # Checkpoint save/load utilities
├── tests/
│   ├── test_dataset.py
│   ├── test_loss.py
│   ├── test_metrics.py
│   └── test_splits.py
├── .gitignore
├── environment.yml               # Conda environment specification
├── pyproject.toml                # Package build configuration
└── README.md
```

## Installation

### Prerequisites 

- Python 3.12 
- CUDA 12.8 (for GPU training - strongly recommended) 
- Conda 

### Steps 

1. Clone the repository: 
 ```bash
git clone https://github.com/kimbielby/DeepZ-Photometric-Redshift-Estimator.git
cd DeepZ-Photometric-Redshift-Estimator
```

2. Create and activate the conda environment: 
```bash
conda env create -f environment.yml
conda activate pzest
```

3. Install the package in editable mode: 
```bash
pip install -e .
```

4. Verify the installation: 
```bash
pytest tests/
```

### Notes 

- **GPU training is strongly recommended.** A single training epoch takes approximately 2-3 minutes on a modern NVIDIA GPU. On CPU this increases to 30-60 minutes per epoch, making full training impractical. 
- NVIDIA GPUs with CUDA 12.8 are supported. AMD GPUs are not supported on Windows. On Linux, AMD GPU support via ROCm may be possible but is not tested - see the [PyTorch ROCm guide](https://pytorch.org/get-started/locally/). 
- If you do not have CUDA 12.8, update the PyTorch index URL in `environment.yml` to match your CUDA version. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/)  for the correct URL 
- If no compatible GPU is detected, PyTorch will automatically fall back to CPU without an error. Training will proceed but will be slow. 
- On Windows, `num_workers` must be set to `0` in `config/default.yaml`. This is the default setting 

## Data

### Dataset 

DeepZ uses the GalaxiesML-Spectra dataset, which pairs Hyper Suprime-Cam (HSC) PDR2 five-band grizy images with DESI DR1 spectroscopic redshifts. The dataset contains 134,533 galaxies with 64x64 pixel images across five photometric bands, covering a redshift range of z = 0 to 4. 

After preprocessing and filtering (removing PSF-classified objects and galaxies with invalid redshifts), 113,245 galaxies are retained for training, validation and testing. 

### Downloading the data 

The dataset is available from Zenodo: 

> GalaxiesML-Spectra: [https://zenodo.org/records/16989593](https://zenodo.org/records/16989593)  

Download `DESI_HSC_64x64_v2.hdf5` and place it in `data/raw/`: 

```
data/
└── raw/
    └── DESI_HSC_64x64_v2.hdf5
```

### Preprocessing  

Preprocessing is a one-time step that filters, stretches and standardises the raw images and saves a clean HDF5 file to `data/processed/`. Run the preprocessing notebook: 

```
notebooks/02_preprocess.ipynb 
``` 

The preprocessing pipeline applies the following steps to each image: 

1. **Filtering** - removes PSF-classified objects and galaxies with redshift ≤ 0 
2. **Arcsinh stretch** - applies `arcsinh(x / sigma_noise)` per band using per-band noise level scale factors to compress the dynamic range and reveal faint galaxy structure 
3. **Per-channel standardisation** - subtracts the per-channel mean and divides by the per-channel standard deviation computed across the full dataset after arcsinh stretching 

The noise scale factors and normalisation constants are derived from the full dataset and stored as constants in `pzest/config.py`. 

### Data Split 

The dataset is split into train, validation and test sets using a fixed random seed for reproducibility: 

| Split | Galaxies | Fraction |
|-------|----------|----------|
| Train | 79,273 | 70% |
| Val | 16,986 | 15% |
| Test | 16,986 | 15% | 

Splits are saved to `data/splits.csv` and reused across all model runs to ensure a consistent held-out test set. 

## Usage

### Command Line 

**Training:**
```bash
python -m pzest.pipelines.train_pipeline 
```

To resume training from the last checkpoint: 
```bash
python -m pzest.pipelines.train_pipeline --resume 
```

To resume training and append history: 
```bash
python -m pzest.pipelines.train_pipeline --resume --append-history
```

**Evaluation:** 
```bash
python -m pzest.pipelines.evaluate_pipeline 
```

### Notebooks 

Alternatively, all steps can be run interactively via Jupyter notebooks in the order below: 

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Dataset exploration and quality checks |
| `02_preprocess.ipynb` | Preprocessing pipeline |
| `03_train_evaluate.ipynb` | Training and evaluation |
| `04_inference.ipynb` | Batch inference on user-provided HDF5 files |
| `05_demo.ipynb` | Visual demo of predicted PDFs on test galaxies |

### Configuration 

All settings are controlled via `config/default.yaml`. Key parameters: 

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.num_bins` | Number of redshift bins | 200 |
| `model.use_magnitudes` | Include magnitude branch | true |
| `train.learning_rate` | Initial learning rate | 1.0e-4 |
| `train.batch_size` | Training batch size | 128 |
| `train.patience` | Early stopping patience | 10 |
| `train.use_weighted_sampling` | Enable redshift-weighted sampling | true |
| `train.sampling_weight_power` | Weighting power (0=none, 0.3=soft, 0.5=sqrt, 1.0=full) | 0.3 |
| `train.label_sigma_bins` | Gaussian label width in bins (0=hard labels) | 2 |

## Models

DeepZ was evaluated across four configurations to study the effect of redshift-weighted sampling on point estimate accuracy and PDF calibration. All models share the same architecture, loss function and hyperparameters - only the sampling strategy and label type differ. 

### Architecture 

The DeepZ model consists of three components: 

- **Image backbone** - a custom Inception CNN with two convolutional stem layers followed by five Inception blocks, producing a 96-dimensional image embedding from a 5 x 64 x 64 input image 
- **Magnitude branch** - a two-layer MLP encoding five HSC grizy CModel magnitudes into a 32-dimensional embedding 
- **Classification head** - a fully connected layer with dropout fusing the 128-dimensional combined embedding into a softmax distribution over 200 redshift bins 

Total trainable parameters: 995,177 

### Model Comparison  

| Model | Sampling power | σ_NMAD | Bias   | Outlier rate | CRPS   | PIT calibration |
|-------|---------------|--------|--------|-------------|--------|-----------------|
| Baseline | 0.0 | 0.0126 | 0.0002 | 0.0070      | 0.9070 | Poor |
| Model 1 | 1.0 | 0.0163 | 0.0070 | 0.0089      | 1.3920 | Good |
| Model 2 | 0.5 | 0.0151 | 0.0080 | 0.0064      | 1.1288 | Good |
| Model 3 | 0.3 | 0.0144 | 0.0073 | 0.0067      | 1.0877 | Good |

### Recommended Model 

**Model 3** (`sampling_weight_power: 0.3`) is recommended for general use. It achieves the best balance between point estimate accuracy and PDF calibration, recovering most of the baseline's sigma_NMAD accuracy whilst eliminating the systematic PDF bias present in the unweighted baseline. 

### Pretrained Checkpoints 

Pretrained checkpoints for all models are available at: 

> [https://zenodo.org/records/19857047](https://zenodo.org/records/19857047) 

Place downloaded checkpoints in `outputs/checkpoints/`.  

## Citation

If you use DeepZ in your research, please cite: 

```bibtex
@software{bielby2026deepz,
  author    = {Bielby, K.},
  title     = {DeepZ: Photometric Redshift Estimation from HSC Galaxy Images},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/kimbielby/DeepZ-Photometric-Redshift-Estimator}
}
``` 

A paper describing the methodology and results is currently under preparation. This citation will be updated with the journal reference upon publication.  

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- Pasquet, J. et al. (2019). *Photometric redshifts from SDSS images using a convolutional neural network*. Astronomy & Astrophysics, 621, A26. [arXiv:1806.06607](https://arxiv.org/abs/1806.06607)

- Jones, E. et al. (2024). *Improving Photometric Redshift Estimation for Cosmology with LSST Using Bayesian Neural Networks*. [DOI:10.3847/1538-4357/ad2070](https://doi.org/10.3847/1538-4357/ad2070)

- Treyer, M. et al. (2024). *CNN photometric redshifts in the SDSS at r ≤ 20*. [DOI:10.1093/mnras/stad3171](https://doi.org/10.1093/mnras/stad3171)

- Saikrishnan, S. et al. (2025). *Multimodal Masked Autoencoder for Galaxy Redshift Estimation*. [arXiv:2510.22527](https://arxiv.org/abs/2510.22527)

- Aihara, H. et al. (2019). *Second data release of the Hyper Suprime-Cam Subaru Strategic Program*. [DOI:10.1093/pasj/psz103](https://doi.org/10.1093/pasj/psz103) 

- GalaxiesML-Spectra dataset. Zenodo. [https://zenodo.org/records/16989593](https://zenodo.org/records/16989593)





