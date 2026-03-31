import argparse
import glob
import multiprocessing
import os
import time
from contextlib import closing
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


parser = argparse.ArgumentParser()
parser.add_argument("-sfp", "--smile_folder_path", help="folder containing split smiles files", required=True)
parser.add_argument("-fn", "--folder_name", help="output folder for Morgan fingerprints", required=True)
parser.add_argument("-tp", "--tot_process", help="number of worker processes", required=True)

io_args = parser.parse_args()
sfp = io_args.smile_folder_path
fn = io_args.folder_name
t_pos = int(io_args.tot_process)

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=1024,
    includeChirality=True,
)


def morgan_fingp(fname):
    input_name = os.path.basename(fname)
    stem = os.path.splitext(input_name)[0]
    if stem.startswith("smile_all_"):
        chunk_id = stem.replace("smile_all_", "", 1)
        output_name = f"morgan_chunk_{chunk_id}.csv"
    else:
        output_name = f"{stem}.csv"
    output_path = os.path.join(fn, output_name)
    with open(output_path, "a") as ref_out:
        with open(fname, "r") as ref_in:
            for line in ref_in:
                smile, compound_id = line.rstrip().split()
                arr = np.zeros((1024,), dtype=np.int8)
                try:
                    mol = Chem.MolFromSmiles(smile)
                    if mol is None:
                        continue
                    fp = MORGAN_GENERATOR.GetFingerprint(mol)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    ref_out.write(",".join([compound_id] + [str(elem) for elem in np.where(arr == 1)[0]]))
                    ref_out.write("\n")
                except Exception:
                    print(line)


files = list(glob.glob(os.path.join(sfp, "*.txt")))

os.makedirs(fn, exist_ok=True)

start_time = time.time()
with closing(Pool(min(multiprocessing.cpu_count(), t_pos))) as pool:
    pool.map(morgan_fingp, files)
print(time.time() - start_time)
