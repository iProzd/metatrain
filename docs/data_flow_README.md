# Metatrain Data Flow Documentation

This document describes the complete data flow in Metatrain from raw input files to
trained models, including the main objects and functions involved in each step.

## Table of Contents

1. [Overview](#overview)
2. [Data Flow Pipeline](#data-flow-pipeline)
3. [Key Data Structures](#key-data-structures)
4. [Step-by-Step Process](#step-by-step-process)
5. [Main Functions and Classes](#main-functions-and-classes)
6. [Example Usage](#example-usage)

## Overview

Metatrain is a CLI for training and evaluating machine learning models for atomistic
systems. The data flow follows this high-level pipeline:

```
Raw Data Files (.xyz, .extxyz, .mts)
    ↓
Data Loaders (readers)
    ↓
Dataset Objects (Dataset/DiskDataset/MemmapDataset)
    ↓
Train/Val/Test Split + Preprocessing
    ↓
DataLoaders with CollateFn (batching)
    ↓
Training Loop (Trainer.train())
    ↓
Model Checkpoint (.ckpt)
    ↓
Export → AtomisticModel (.pt)
```

## Data Flow Pipeline

### 1. Input Data Formats

Metatrain supports three main input formats:

- **XYZ/Extended XYZ**: Standard atomic structure format read via ASE
- **Metatensor (.mts)**: Native metatensor format with hierarchical tensor data
- **Memory-mapped datasets**: For large datasets on HPC systems

**Key Files:**
- `src/metatrain/utils/data/readers/ase.py` - ASE reader for XYZ files
- `src/metatrain/utils/data/readers/metatensor.py` - Metatensor reader
- `src/metatrain/utils/data/readers/readers.py` - Main reader dispatcher

### 2. Data Loading

**Function:** `read_systems(filename, reader=None)` 
- **Location:** `src/metatrain/utils/data/readers/readers.py`
- **Input:** File path and optional reader name
- **Output:** List of `System` objects (from metatomic.torch)
- **Purpose:** Load atomic structures from files

**Function:** `read_targets(conf)`
- **Location:** `src/metatrain/utils/data/readers/readers.py`
- **Input:** Configuration dictionary with target specifications
- **Output:** Tuple of (Dict[str, List[TensorMap]], Dict[str, TargetInfo])
- **Purpose:** Load target properties (energy, forces, stress, etc.)

### 3. Dataset Creation

**Function:** `get_dataset(options)`
- **Location:** `src/metatrain/utils/data/get_dataset.py`
- **Input:** Configuration dictionary with system and target paths
- **Output:** Tuple of (Dataset, target_info, extra_data_info)
- **Purpose:** Create a unified dataset object from systems and targets

**Dataset Types:**
- `Dataset`: In-memory dataset for small datasets (<10k structures)
- `DiskDataset`: Memory-mapped from .zip files for large datasets
- `MemmapDataset`: Memory-mapped from directories for parallel filesystems

### 4. Data Preprocessing

Several preprocessing steps are applied before training:

#### a. Additive Models (Baseline Removal)
- **CompositionModel**: Removes per-element energy contributions
- **ZBL**: Removes short-range ionic repulsion baseline
- **Location:** `src/metatrain/utils/additive/`
- **Function:** `get_remove_additive_transform()`

#### b. Scaling (Normalization)
- **Scaler**: Normalizes targets using z-score: `(y - mean) / std`
- **Location:** `src/metatrain/utils/scaler/scaler.py`
- **Methods:** 
  - `train_model()`: Compute mean/std from training data
  - `forward()`: Apply scaling transformation
  - `get_remove_scale_transform()`: Returns preprocessing function

#### c. Neighbor Lists
- **Purpose:** Pre-compute atomic neighborhoods for descriptor calculation
- **Function:** `get_system_with_neighbor_lists_transform()`
- **Location:** `src/metatrain/utils/neighbor_lists.py`

### 5. Batching and Data Loading

**Class:** `CollateFn`
- **Location:** `src/metatrain/utils/data/dataset.py`
- **Purpose:** Collate individual samples into batches
- **Process:**
  1. Apply preprocessing transforms (neighbor lists, additive removal, scaling)
  2. Serialize Systems and TensorMaps to binary buffers
  3. Concatenate into single blob with metadata (sizes, names)
- **Output:** `(blob, system_sizes, target_names, target_sizes, extra_names,
  extra_sizes)`

**Function:** `unpack_batch(batch)`
- **Location:** `src/metatrain/utils/data/dataset.py`
- **Purpose:** Deserialize batch in model forward pass
- **Returns:** Systems and TensorMaps (targets + extra_data)

**Class:** `CombinedDataLoader`
- **Location:** `src/metatrain/utils/data/combine_dataloaders.py`
- **Purpose:** Combine multiple dataloaders from different datasets
- **Features:** Shuffles batches across datasets during training

### 6. Training Pipeline

**Entry Point:** `train_model(options)`
- **Location:** `src/metatrain/cli/train.py`
- **Process:**
  1. Load architecture dynamically via `import_architecture()`
  2. Create Dataset objects for train/val/test splits
  3. Initialize DatasetInfo with atomic types and target metadata
  4. Create Model and Trainer instances
  5. Execute training loop via `trainer.train()`
  6. Save checkpoint and export model

**Architecture Pattern:**
Each architecture (e.g., `soap_bpnn`, `pet`, `gap`) implements:
- `ModelInterface`: Base class for models
- `TrainerInterface`: Base class for trainers
- Located in: `src/metatrain/{architecture_name}/`

### 7. Loss Computation

**Class:** `LossAggregator`
- **Location:** `src/metatrain/utils/loss.py`
- **Purpose:** Combine multiple loss functions for different targets
- **Supported Losses:**
  - `TensorMapMSELoss`: Mean Squared Error
  - `TensorMapMAELoss`: Mean Absolute Error
  - `TensorMapHuberLoss`: Huber loss

### 8. Model Export

**Function:** `export_model(model_path, options)`
- **Location:** `src/metatrain/cli/export.py`
- **Purpose:** Convert trained model to TorchScript format
- **Output:** AtomisticModel (.pt) compatible with MD engines (LAMMPS, ASE, i-PI)

## Key Data Structures

### System (from metatomic.torch)
Represents an atomic structure with:
- `types`: Integer atomic types
- `positions`: Atomic coordinates (Nx3)
- `cell`: Simulation cell (3x3)
- `pbc`: Periodic boundary conditions
- Optional: neighbor lists, forces, stress

### TensorMap (from metatensor.torch)
Hierarchical tensor representation of properties:
- `keys`: Metadata labels for blocks
- `blocks`: List of TensorBlock objects with:
  - `values`: Actual tensor data
  - `samples`: Labels for each sample
  - `components`: Labels for tensor components (e.g., xyz for vectors)
  - `properties`: Labels for properties

### TargetInfo
- **Location:** `src/metatrain/utils/data/target_info.py`
- **Contents:**
  - `layout`: Empty TensorMap defining structure
  - `quantity`: Physical quantity (e.g., "energy", "dipole")
  - `unit`: Unit of measurement
  - `per_atom`: Whether property is per-atom or global
  - `gradients`: Names of gradients (e.g., "forces" for energy)

### DatasetInfo
- **Location:** `src/metatrain/utils/data/dataset.py`
- **Contents:**
  - `length_unit`: Unit of atomic structures
  - `atomic_types`: List of unique atomic types in dataset
  - `targets`: Dict[str, TargetInfo] - metadata for each target
  - `extra_data`: Additional per-sample data

## Step-by-Step Process

### Step 1: Load Raw Data

```python
from metatrain.utils.data.readers import read_systems, read_targets

# Read atomic structures
systems = read_systems("data.xyz", reader="ase")

# Read target properties
targets_conf = {
    "energy": {
        "key": "energy",
        "read_from": "data.xyz",
        "reader": "ase",
        "forces": {"read_from": "data.xyz"},
    }
}
targets, target_info = read_targets(targets_conf)
```

### Step 2: Create Dataset

```python
from metatrain.utils.data import Dataset, get_dataset

# From systems and targets
dataset = Dataset.from_dict({"system": systems, **targets})

# Or using get_dataset
dataset, target_info, extra_data_info = get_dataset(options)
```

### Step 3: Split into Train/Val/Test

```python
from metatrain.utils.data import _train_test_random_split

train_dataset, val_dataset, test_dataset = _train_test_random_split(
    dataset,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
)
```

### Step 4: Create DatasetInfo

```python
from metatrain.utils.data.dataset import DatasetInfo, get_atomic_types

atomic_types = get_atomic_types([train_dataset, val_dataset])
dataset_info = DatasetInfo(
    length_unit="angstrom",
    atomic_types=atomic_types,
    targets=target_info,
)
```

### Step 5: Initialize Model and Trainer

```python
from metatrain.utils.architectures import import_architecture

# Load architecture
architecture = import_architecture("soap_bpnn")
Model = architecture.__model__
Trainer = architecture.__trainer__

# Initialize
model = Model(model_hypers, dataset_info)
trainer = Trainer(training_hypers)
```

### Step 6: Train

```python
# Train the model
trainer.train(
    model=model,
    devices=[device],
    train_datasets=[train_dataset],
    val_datasets=[val_dataset],
    checkpoint_dir="outputs/checkpoints/",
)
```

### Step 7: Export

```python
# Save checkpoint
trainer.save_checkpoint(model, "model.ckpt")

# Export to AtomisticModel
atomistic_model = model.export()
atomistic_model.save("model.pt")
```

## Main Functions and Classes

### Data Loading and Reading

| Component | Location | Purpose |
|-----------|----------|---------|
| `read_systems()` | `utils/data/readers/readers.py` | Load atomic structures from files |
| `read_targets()` | `utils/data/readers/readers.py` | Load target properties from files |
| `read_extra_data()` | `utils/data/readers/readers.py` | Load additional data |
| `get_dataset()` | `utils/data/get_dataset.py` | Create unified dataset from configs |

### Dataset Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `Dataset` | `utils/data/dataset.py` | In-memory dataset |
| `DiskDataset` | `utils/data/dataset.py` | Memory-mapped dataset from .zip |
| `MemmapDataset` | `utils/data/dataset.py` | Memory-mapped dataset from directory |
| `DatasetInfo` | `utils/data/dataset.py` | Dataset metadata container |
| `TargetInfo` | `utils/data/target_info.py` | Target property metadata |

### Preprocessing

| Component | Location | Purpose |
|-----------|----------|---------|
| `Scaler` | `utils/scaler/scaler.py` | Target normalization |
| `CompositionModel` | `utils/additive/composition.py` | Remove composition baseline |
| `ZBL` | `utils/additive/zbl.py` | Remove ionic repulsion baseline |
| `get_system_with_neighbor_lists()` | `utils/neighbor_lists.py` | Add neighbor lists |

### Batching and Data Loading

| Component | Location | Purpose |
|-----------|----------|---------|
| `CollateFn` | `utils/data/dataset.py` | Batch collation with transforms |
| `unpack_batch()` | `utils/data/dataset.py` | Deserialize batch |
| `CombinedDataLoader` | `utils/data/combine_dataloaders.py` | Combine multiple dataloaders |

### Training

| Component | Location | Purpose |
|-----------|----------|---------|
| `train_model()` | `cli/train.py` | Main training entry point |
| `import_architecture()` | `utils/architectures.py` | Load architecture dynamically |
| `ModelInterface` | `utils/abc.py` | Base class for models |
| `TrainerInterface` | `utils/abc.py` | Base class for trainers |

### Loss Functions

| Component | Location | Purpose |
|-----------|----------|---------|
| `LossAggregator` | `utils/loss.py` | Combine multiple losses |
| `TensorMapMSELoss` | `utils/loss.py` | Mean Squared Error |
| `TensorMapMAELoss` | `utils/loss.py` | Mean Absolute Error |
| `TensorMapHuberLoss` | `utils/loss.py` | Huber loss |

### Model Export

| Component | Location | Purpose |
|-----------|----------|---------|
| `export_model()` | `cli/export.py` | Export to AtomisticModel |
| `AtomisticModel` | metatomic library | TorchScript model wrapper |

## Example Usage

For a complete working example that demonstrates the data flow with a mock model, see:
- [examples/data_flow_example.py](../examples/data_flow_example.py)

This example creates synthetic data and runs through the entire training pipeline with a
simple mock model.

## Key Transformations

1. **System → Features**: Architecture-specific descriptors (e.g., SOAP, embeddings)
2. **Target Normalization**: `y_normalized = (y - mean) / std`
3. **Additive Removal**: `y' = y - composition_baseline - zbl_baseline`
4. **Batching**: Serialize to tensors for GPU processing
5. **Loss Computation**: MSE/MAE on predictions vs normalized targets
6. **Export**: TorchScript compilation with metadata embedding

## Additional Resources

- [Main README](../README.md)
- [Contributing Guide](../CONTRIBUTING.rst)
- [Example Gallery](https://metatensor.github.io/metatrain/latest/examples/)
- [API Documentation](https://metatensor.github.io/metatrain/latest/)
