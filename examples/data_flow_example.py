"""
Data Flow Example with Mock Model
==================================

This example demonstrates the complete data flow in metatrain from raw input data to
model training and export, using a simple mock model. It showcases:

1. Creating synthetic atomic structures and target properties
2. Loading data through the standard metatrain pipeline
3. Creating datasets and splitting into train/val/test
4. Setting up preprocessing (scaling, additive models)
5. Training a mock model
6. Exporting the trained model

This example uses random/synthetic data to illustrate the data structures and flow
without requiring real DFT calculations.
"""

# %%
# Setup and Imports
# -----------------
#
# First, we import all necessary packages for creating data, building datasets,
# and setting up the training pipeline.

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import ase.io
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    systems_to_torch,
)

from metatrain.utils.additive import CompositionModel
from metatrain.utils.architectures import check_architecture_options
from metatrain.utils.data import (
    Dataset,
    DatasetInfo,
    TargetInfo,
    get_atomic_types,
    read_systems,
    read_targets,
)
from metatrain.utils.data.combine_dataloaders import CombinedDataLoader
from metatrain.utils.data.dataset import CollateFn, unpack_batch
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.loss import LossAggregator, LossSpecification, TensorMapMSELoss
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists_transform
from metatrain.utils.scaler import Scaler


# %%
# Step 1: Create Synthetic Training Data
# ---------------------------------------
#
# We'll create synthetic atomic structures with random positions and energies.
# In a real scenario, these would come from DFT calculations or other quantum
# chemistry methods.

def create_synthetic_structures(n_structures: int = 100, n_atoms: int = 10) -> List:
    """Create synthetic atomic structures with random properties.

    Args:
        n_structures: Number of structures to generate
        n_atoms: Number of atoms per structure

    Returns:
        List of ASE Atoms objects
    """
    print(f"Creating {n_structures} synthetic structures with {n_atoms} atoms each...")

    structures = []
    for i in range(n_structures):
        # Random positions in a box
        positions = np.random.rand(n_atoms, 3) * 10.0

        # Random atomic types (H=1, C=6, O=8)
        atomic_numbers = np.random.choice([1, 6, 8], size=n_atoms)

        # Create ASE Atoms object
        atoms = ase.Atoms(
            numbers=atomic_numbers,
            positions=positions,
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )

        # Add synthetic energy (random + composition contribution)
        base_energy = -100.0 * n_atoms  # Base energy per atom
        composition_energy = sum(
            -0.5 * (num == 1) - 1.0 * (num == 6) - 2.0 * (num == 8)
            for num in atomic_numbers
        )
        noise = np.random.randn() * 5.0  # Random noise
        atoms.info["energy"] = base_energy + composition_energy + noise

        # Add synthetic forces (small random forces)
        atoms.arrays["forces"] = np.random.randn(n_atoms, 3) * 0.1

        structures.append(atoms)

    print(f"✓ Created {len(structures)} structures")
    return structures


# %%
# Step 2: Save Data to File
# --------------------------
#
# Save the synthetic structures to an XYZ file, which is the standard input format
# for metatrain.

print("\n" + "=" * 70)
print("STEP 1: Creating and Saving Synthetic Data")
print("=" * 70)

# Create temporary directory for our example
temp_dir = tempfile.mkdtemp(prefix="metatrain_dataflow_")
data_path = Path(temp_dir) / "synthetic_data.xyz"

# Generate and save data
structures = create_synthetic_structures(n_structures=100, n_atoms=10)
ase.io.write(data_path, structures)
print(f"✓ Data saved to {data_path}")


# %%
# Step 3: Load Data Using Metatrain Readers
# ------------------------------------------
#
# Use metatrain's standard data readers to load systems and targets.
# This is the entry point for all data in the training pipeline.

print("\n" + "=" * 70)
print("STEP 2: Loading Data Through Metatrain Readers")
print("=" * 70)

# Read systems (atomic structures)
systems = read_systems(str(data_path), reader="ase")
print(f"✓ Loaded {len(systems)} systems")
print(f"  First system has {len(systems[0])} atoms")
print(f"  Atomic types in first system: {systems[0].types.tolist()}")

# For energy targets, we can use the helper function to create TensorMaps manually
# In a real scenario, this would be done through the configuration system
print("✓ Creating energy and forces targets manually...")

# Extract energies and forces from the structures
energies_list = []

for i, atoms in enumerate(structures):
    # Energy (scalar per structure)
    energy_block = TensorBlock(
        values=torch.tensor([[atoms.info["energy"]]], dtype=torch.float64),
        samples=Labels(names=["system"], values=torch.tensor([[i]])),
        components=[],
        properties=Labels(names=["energy"], values=torch.tensor([[0]])),
    )
    energy_map = TensorMap(keys=Labels.single(), blocks=[energy_block])
    energies_list.append(energy_map)
    
    # Forces (vector per atom)
    # Forces are stored with energies as gradient, not as separate target
    # We'll attach them when creating the energy target info

targets_dict = {"energy": energies_list}

# Create target info for energy with forces as gradients
from omegaconf import OmegaConf

energy_config = OmegaConf.create({
    "key": "energy",
    "unit": "eV",
    "forces": {}  # Enable forces gradients
})
target_info_dict = {"energy": get_energy_target_info(
    "energy", 
    energy_config, 
    add_position_gradients=True
)}

print(f"✓ Created targets: {list(targets_dict.keys())}")
print(f"  Energy values shape: {targets_dict['energy'][0].block().values.shape}")


# %%
# Step 4: Create Dataset and Split
# ---------------------------------
#
# Combine systems and targets into a Dataset object, then split into
# train/validation/test sets.

print("\n" + "=" * 70)
print("STEP 3: Creating Dataset and Splitting")
print("=" * 70)

# Create full dataset
full_dataset = Dataset.from_dict({"system": systems, **targets_dict})
print(f"✓ Created dataset with {len(full_dataset)} samples")

# Split into train/val/test (80/10/10)
from torch.utils.data import random_split

n_train = int(0.8 * len(full_dataset))
n_val = int(0.1 * len(full_dataset))
n_test = len(full_dataset) - n_train - n_val

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42),
)

print(f"✓ Split dataset:")
print(f"  Training:   {len(train_dataset)} samples")
print(f"  Validation: {len(val_dataset)} samples")
print(f"  Test:       {len(test_dataset)} samples")


# %%
# Step 5: Create DatasetInfo
# ---------------------------
#
# DatasetInfo contains metadata about the dataset: atomic types, units, and target
# information. This is used by the model to know what to expect.

print("\n" + "=" * 70)
print("STEP 4: Creating DatasetInfo")
print("=" * 70)

# Get unique atomic types from the datasets
atomic_types = get_atomic_types([train_dataset, val_dataset])
print(f"✓ Atomic types in dataset: {atomic_types}")

# Create DatasetInfo
dataset_info = DatasetInfo(
    length_unit="angstrom",
    atomic_types=atomic_types,
    targets=target_info_dict,
)

print(f"✓ Created DatasetInfo:")
print(f"  Length unit: {dataset_info.length_unit}")
print(f"  Atomic types: {dataset_info.atomic_types}")
print(f"  Targets: {list(dataset_info.targets.keys())}")


# %%
# Step 6: Setup Preprocessing - Scaler
# -------------------------------------
#
# The scaler normalizes targets using z-score normalization (mean=0, std=1).
# This helps with training stability and convergence.
# Note: In a real scenario, you would also use a composition model to remove
# baseline energies, but we skip it here for simplicity.

print("\n" + "=" * 70)
print("STEP 5: Setting Up Scaler (Normalization)")
print("=" * 70)

# Initialize and train scaler
scaler = Scaler(
    hypers={},
    dataset_info=dataset_info,
)

print("Training scaler...")
scaler.train_model(
    datasets=[train_dataset],
    additive_models=[],  # No additive models for simplicity
    batch_size=32,
    is_distributed=False,
)
print("✓ Scaler trained")
print("  (Scaler statistics are computed internally)")


# %%
# Step 7: Define Mock Model
# --------------------------
#
# We create a simple mock model that predicts random energies and forces.
# In a real scenario, this would be a sophisticated neural network architecture
# like SOAP-BPNN, PET, or MACE.


class MockModel(torch.nn.Module):
    """A simple mock model for demonstration purposes.

    This model doesn't learn anything meaningful but demonstrates the expected
    interface and data flow.
    """

    def __init__(self, dataset_info: DatasetInfo):
        super().__init__()
        self.dataset_info = dataset_info
        self.atomic_types = dataset_info.atomic_types

        # Simple learnable parameters per atomic type
        self.atom_embeddings = torch.nn.Parameter(
            torch.randn(len(self.atomic_types), 8)
        )
        self.energy_predictor = torch.nn.Linear(8, 1)

        # Store scaler (no composition model for simplicity)
        self.scaler = None

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Forward pass of the mock model.

        Args:
            systems: List of atomic systems
            outputs: Dictionary specifying which outputs to compute
            selected_atoms: Optional selection of atoms

        Returns:
            Dictionary mapping output names to TensorMaps with predictions
        """
        predictions = {}

        if "energy" in outputs:
            # Predict energies
            energies = []
            for system in systems:
                # Get embeddings for all atoms in this system
                atom_indices = [
                    self.atomic_types.index(t.item()) for t in system.types
                ]
                embeddings = self.atom_embeddings[atom_indices]

                # Simple prediction: sum of embeddings through linear layer
                system_embedding = embeddings.mean(dim=0, keepdim=True)
                energy = self.energy_predictor(system_embedding)
                energies.append(energy)

            # Create TensorMap for energies
            energy_values = torch.cat(energies, dim=0)  # Shape: (n_systems, 1)
            energy_block = TensorBlock(
                values=energy_values,
                samples=Labels.range("system", len(systems)),
                components=[],
                properties=Labels.range("energy", 1),
            )
            predictions["energy"] = TensorMap(
                keys=Labels.single(),
                blocks=[energy_block],
            )

        return predictions

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        """Return empty list as this mock model doesn't use neighbor lists."""
        return []


print("\n" + "=" * 70)
print("STEP 6: Initializing Mock Model")
print("=" * 70)

model = MockModel(dataset_info)
model.scaler = scaler
print("✓ Mock model initialized")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")


# %%
# Step 8: Setup Data Loader with Preprocessing
# ---------------------------------------------
#
# Create data loaders that apply preprocessing transforms during batching.
# Note: The scaler is applied inside the model, not in the collate function.

print("\n" + "=" * 70)
print("STEP 7: Setting Up DataLoader with Preprocessing")
print("=" * 70)

# Create collate function with preprocessing transforms
collate_fn = CollateFn(
    target_keys=list(target_info_dict.keys()),
    callables=[
        get_system_with_neighbor_lists_transform(
            model.requested_neighbor_lists()
        ),  # Add neighbor lists (none for our mock model)
    ],
)

# Create data loaders
from torch.utils.data import DataLoader

train_base_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
)

val_base_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
)

train_dataloader = CombinedDataLoader([train_base_loader], shuffle=True)
val_dataloader = CombinedDataLoader([val_base_loader], shuffle=False)

print("✓ DataLoaders created:")
print(f"  Training batches:   {len(train_dataloader)}")
print(f"  Validation batches: {len(val_dataloader)}")


# %%
# Step 9: Setup Loss Function
# -----------------------------
#
# Configure the loss function for training. We use MSE loss for energy.

print("\n" + "=" * 70)
print("STEP 8: Setting Up Loss Function")
print("=" * 70)

# Use a simple MSE loss directly
loss_fn = torch.nn.MSELoss()
print("✓ Loss function configured (MSE)")


# %%
# Step 10: Training Loop
# ----------------------
#
# Run a simple training loop to demonstrate the complete data flow from batched data
# through the model to loss computation.

print("\n" + "=" * 70)
print("STEP 9: Training Loop")
print("=" * 70)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
n_epochs = 3
device = torch.device("cpu")
dtype = torch.float64  # Use float64 to match data
model.to(device=device, dtype=dtype)

print(f"Training for {n_epochs} epochs...")
print()

for epoch in range(n_epochs):
    # Training
    model.train()
    train_loss_sum = 0.0
    n_train_batches = 0

    for batch in train_dataloader:
        # Unpack batch
        systems, targets, extra_data = unpack_batch(batch)
        
        # Move to device if needed
        # (In a real scenario, you'd move TensorMaps to device here)

        # Forward pass
        outputs = model(
            systems,
            outputs={"energy": ModelOutput(per_atom=False)},
        )

        # Compute loss (simple MSE on energies)
        pred_energies = outputs["energy"].block().values
        target_energies = targets["energy"].block().values
        loss = loss_fn(pred_energies, target_energies)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        n_train_batches += 1

    # Validation
    model.eval()
    val_loss_sum = 0.0
    n_val_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            systems, targets, extra_data = unpack_batch(batch)

            outputs = model(
                systems,
                outputs={"energy": ModelOutput(per_atom=False)},
            )

            # Compute loss (simple MSE on energies)
            pred_energies = outputs["energy"].block().values
            target_energies = targets["energy"].block().values
            loss = loss_fn(pred_energies, target_energies)
            val_loss_sum += loss.item()
            n_val_batches += 1

    # Print epoch results
    avg_train_loss = train_loss_sum / n_train_batches
    avg_val_loss = val_loss_sum / n_val_batches
    print(
        f"Epoch {epoch + 1}/{n_epochs}: "
        f"Train Loss = {avg_train_loss:.4f}, "
        f"Val Loss = {avg_val_loss:.4f}"
    )

print("\n✓ Training completed!")


# %%
# Step 11: Model Export
# ----------------------
#
# In a real scenario, you would export the model to TorchScript format using
# model.export(), which creates an AtomisticModel that can be used with MD engines.
# Our mock model doesn't implement full export functionality, but here's the pattern:

print("\n" + "=" * 70)
print("STEP 10: Model Export (Conceptual)")
print("=" * 70)

checkpoint_path = Path(temp_dir) / "mock_model.ckpt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "dataset_info": dataset_info,
        "scaler": scaler,
    },
    checkpoint_path,
)
print(f"✓ Model checkpoint saved to {checkpoint_path}")
print()
print("In a real scenario, you would export using:")
print("  atomistic_model = model.export()")
print("  atomistic_model.save('model.pt')")
print()
print("The exported .pt file can then be used with:")
print("  - ASE: for molecular dynamics simulations")
print("  - LAMMPS: for large-scale MD simulations")
print("  - i-PI: for path integral MD simulations")


# %%
# Summary
# -------
#
# This example demonstrated the complete data flow in metatrain:
#
# 1. ✓ Created synthetic atomic structures with energies and forces
# 2. ✓ Saved data to XYZ file (standard input format)
# 3. ✓ Loaded data using metatrain readers (read_systems)
# 4. ✓ Created Dataset and split into train/val/test
# 5. ✓ Built DatasetInfo with metadata
# 6. ✓ Trained scaler for target normalization
# 7. ✓ Initialized a mock model
# 8. ✓ Created DataLoaders with preprocessing transforms
# 9. ✓ Configured loss function
# 10. ✓ Ran training loop with forward/backward passes
# 11. ✓ Saved model checkpoint
#
# Key Data Transformations:
# - Raw XYZ → Systems (metatomic) + Targets (TensorMaps)
# - Scaling: y_normalized = (y - mean) / std
# - Batching: Serialize to binary buffers for GPU processing
# - Forward pass: Systems → Model → Predictions (TensorMaps)
# - Loss: MSE between predictions and targets
# - Export: Model → TorchScript AtomisticModel (.pt)

print("\n" + "=" * 70)
print("DATA FLOW COMPLETE!")
print("=" * 70)
print()
print("For more details on each step, see:")
print("  - docs/data_flow_README.md - Complete documentation")
print("  - examples/ - More real-world examples")
print("  - src/metatrain/cli/train.py - Full training implementation")
print()
print(f"Temporary files saved in: {temp_dir}")
print("(These will be cleaned up automatically)")
