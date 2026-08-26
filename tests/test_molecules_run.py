"""Pytest tests for MoleculeResolver: registry, raw geometry, SMILES (RDKit),
common-name (PubChem network lookup), and batch resolution.

Run:
    pytest tests/test_molecules_run.py                  # everything, incl. network tests
    pytest tests/test_molecules_run.py -m "not network"  # skip PubChem lookups
"""
import os
import sys

sys.path.insert(0, os.path.abspath("./build"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.api.molecule_resolver import (MoleculeResolver, MoleculeTooBigError, ResolutionError)

try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


@pytest.fixture(scope="module")
def resolver():
    return MoleculeResolver(max_qubits=20, allow_network=True, cache_dir=".pubchem_cache")


@pytest.fixture(scope="module")
def big_resolver():
    """Ethanol needs 36 qubits after active-space reduction -- exceeds the
    max_qubits=20 used everywhere else (deliberately, so test_too_big_rejected
    has something to reject). Needs its own resolver instance."""
    return MoleculeResolver(max_qubits=40, allow_network=True, cache_dir=".pubchem_cache")


# ---------- deterministic: local registry, no network ----------

def test_registry_lookup_lih(resolver):
    result = resolver.resolve("LiH")
    assert result.source == "registry"
    assert result.name == "LiH"
    assert result.estimated_qubits > 0
    assert result.geometry


def test_registry_lookup_beh2(resolver):
    result = resolver.resolve("BeH2")
    assert result.source == "registry"
    assert result.estimated_qubits > 0


def test_force_no_freeze(resolver):
    """freeze_core=False must be honored and must not reduce the active space."""
    result = resolver.resolve("BeH2", freeze_core=False)
    assert result.freeze_core is False
    assert result.active_electrons == result.total_electrons


def test_too_big_rejected(resolver):
    """CO2 (24 qubits per the registry) exceeds this resolver's max_qubits=20,
    and resolves via the local registry -- no network needed, so this can't
    flake on an external PubChem lookup the way the original "adenine" case did.
    """
    with pytest.raises(MoleculeTooBigError):
        resolver.resolve("CO2")


# ---------- deterministic: raw geometry string, no network ----------

def test_raw_geometry(resolver):
    result = resolver.resolve("C 0 0 0; H 0 0 1.09")
    assert result.source == "raw"
    assert result.estimated_qubits > 0


# ---------- network: PubChem common-name lookup ----------

@pytest.mark.network
def test_common_name_water(resolver):
    result = resolver.resolve("water")
    assert result.source == "pubchem"
    assert result.estimated_qubits > 0
    assert result.geometry


@pytest.mark.network
def test_common_name_ethanol(big_resolver):
    result = big_resolver.resolve("ethanol")
    assert result.source == "pubchem"
    assert result.estimated_qubits > 0


# ---------- SMILES: requires RDKit (optional dependency) ----------

@pytest.mark.skipif(not HAS_RDKIT, reason="rdkit not installed")
def test_smiles_ethanol(big_resolver):
    result = big_resolver.resolve("CCO")
    assert result.source == "smiles"
    assert result.estimated_qubits > 0


@pytest.mark.skipif(not HAS_RDKIT, reason="rdkit not installed")
def test_smiles_methane(resolver):
    result = resolver.resolve("C")
    assert result.source == "smiles"
    assert result.estimated_qubits > 0


# ---------- batch resolution (registry-only molecules, deterministic) ----------

def test_batch_resolution(resolver):
    batch = resolver.resolve_batch(["H2", "LiH", "BeH2"])
    assert batch["H2"] is not None
    assert batch["LiH"] is not None
    assert batch["BeH2"] is not None
    for name, result in batch.items():
        assert result.source == "registry"
        assert result.estimated_qubits > 0
