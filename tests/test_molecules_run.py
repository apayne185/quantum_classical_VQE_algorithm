import sys
import os
sys.path.insert(0, os.path.abspath("./build"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.molecule_resolver import (MoleculeResolver, MoleculeTooBigError, ResolutionError)   
# sys.path.insert(0, os.path.abspath("."))
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


resolver = MoleculeResolver(max_qubits=20,allow_network=True, cache_dir=".pubchem_cache")

test_cases = [
    ("Registry lookup","LiH",{}),
    ("Registry lookup", "BeH2", {}),
    ("Common name","water", {}),                # PubChem
    ("Common name","ethanol",{}),               # PubChem
    ("SMILES","CCO",{}),                 # ethanol RDKit
    ("SMILES","C", {}),                     #methane
    ("Raw geometry","C 0 0 0; H 0 0 1.09", {}),
    ("Too big (rejected)", "adenine",{}),            # >20 
    ("Force no-freeze","BeH2",{"freeze_core": False}),    
]

print("MOLECULE RESOLVER tests")

for label, inp, kwargs in test_cases: 
    print(f"\n[{label}] Input: '{inp}'")
    try:
        result = resolver.resolve(inp, **kwargs)
        print(f"Source: {result.source}")
        print(f"Geometry (first): {result.geometry[:60]}...")
        print(f"Total electrons:{result.total_electrons}")
        print(f"Active electrons: {result.active_electrons}")
        print(f"Estimated qubits: {result.estimated_qubits}")
        print(f"Freeze core:{result.freeze_core}")    

        if result.warnings:
            for w in result.warnings:
                print(f"Warning:{w}")     


    except MoleculeTooBigError as e:
        print(f"[GATED] {e}")
    except ResolutionError as e:
        print(f"[FAILED] {e}")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}:{e}")    


print("Batch resolution example ")
batch= resolver.resolve_batch(["H2","LiH","aspirin","BeH2"])   

for name, res in batch.items():
    if res:
        print(f"{name:12s}: {res.estimated_qubits:3d} qubits, source={res.source}")
    else:
        print(f"{name:12s}: failed or too big ")   

