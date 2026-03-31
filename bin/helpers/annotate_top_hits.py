#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings, CalcNumRings, CalcTPSA


def mol_from_smiles(smiles: str | None):
    if smiles is None:
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def drug_like_label(props: dict[str, float]) -> str:
    passes = (
        250 <= props["MW"] <= 550
        and 1 <= props["cLogP"] <= 5
        and props["HBD"] <= 5
        and props["HBA"] <= 10
        and props["TPSA"] <= 140
        and props["RotB"] <= 10
        and props["RingCount"] <= 6
        and props["AromaticRings"] <= 4
        and 20 <= props["HeavyAtoms"] <= 50
        and -1 <= props["FormalCharge"] <= 1
    )
    return "Yes" if passes else "No"


def calc_props(mol):
    props = {
        "MW": Descriptors.MolWt(mol),
        "cLogP": Crippen.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "TPSA": CalcTPSA(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "RingCount": CalcNumRings(mol),
        "AromaticRings": CalcNumAromaticRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "FormalCharge": Chem.GetFormalCharge(mol),
    }
    props["Drug_like"] = drug_like_label(props)
    return props


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-hits-table", required=True)
    parser.add_argument("--boltz-table", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    top_hits = pd.read_csv(args.top_hits_table)
    boltz = pd.read_csv(args.boltz_table, sep=r"\s+|\t", engine="python")
    boltz = boltz.rename(
        columns={
            "affinity_pred_log10_IC50": "Boltz2_pred_log10IC50",
            "affinity_probability_binary": "Boltz2_pred_prob",
            "confidence_score": "Boltz2_confidence_score",
            "ptm": "Boltz2_ptm",
            "iptm": "Boltz2_iptm",
            "ligand_iptm": "Boltz2_ligand_iptm",
        }
    )

    merged = top_hits.merge(boltz, on="CID", how="left")

    prop_rows = []
    valid_smiles = []
    for smiles in merged["SMILES"]:
        mol = mol_from_smiles(smiles)
        if mol is None:
            prop_rows.append({})
            valid_smiles.append(False)
        else:
            prop_rows.append(calc_props(mol))
            valid_smiles.append(True)

    props = pd.DataFrame(prop_rows)
    output = pd.concat([merged, props], axis=1)
    output["valid_smiles"] = valid_smiles
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
