import re
import os
import json
import glob
import shutil
import hashlib
from pathlib import Path
from functools import cmp_to_key
from distutils.version import StrictVersion

REGEX_DIR = r"v-([^-]+)-([^-]+)-([^-]+)-check-([^-]+)-of-([^-]+)"
REGEX_JPG = r"prompt-([^-]+)"
OUT_DIR = './merged'

PROMPTS = {
    '3.3.0': 3 * (
'Photograph in pantry of a name-brand plastic spice jar with a small square paper label for the {spice}',
    )
}

def main():
    out_dir = Path(OUT_DIR)
    try:
        shutil.rmtree(out_dir)
    except: pass
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = glob.glob('./*/images/index.json') 
    indexed = []

    for (i, index) in enumerate(indices):
        image_dir = Path(index).parent
        run_name = image_dir.parent.name
        result = re.search(REGEX_DIR, run_name)
        groups = () if result is None else result.groups(1)
        v = "0.0.0" if len(groups) < 3 else '.'.join(groups[:3])
        reps = "0/0" if len(groups) < 5 else '/'.join(groups[-2:])
        run_info = { "v": v, "reps": reps, "run": i }

        indexed.append({
            "v": v,
            "reps": reps,
            "index": index,
            "run_info": run_info
        })

    indexed.sort(key=cmp_to_key(compare_index))

    merged = []
    prompts = ()

    for meta in indexed:
        v = meta["v"]
        index = meta["index"]
        run_info = meta["run_info"]
        image_dir = Path(index).parent
        i_str = to_hash(str(image_dir))
        prompts += PROMPTS[v]
        p_offset = len(prompts) - len(PROMPTS[v])
        count_prompt = lambda p: int(p) + p_offset

        meta_data = []
        with open(index, 'r') as rf:
            meta_data = json.load(rf)

        for (j, image_meta) in enumerate(meta_data):
            j_str = str(j).zfill(7)

            file_name = image_meta["file_name"]

            # Count the new prompt index
            result = re.search(REGEX_JPG, file_name)
            groups = () if result is None else result.groups(1)
            prompt = "" if len(groups) < 1 else count_prompt(groups[0])
            # Rename the file and store new prompt index
            new_name = f'run-{i_str}-image-{j_str}.jpg'
            image_meta["file_name"] = new_name
            image_meta["prompt"] = prompt
            print(file_name, '->' , image_meta["file_name"])
            image_meta.update(run_info)

            # Copy the image file into the merged directory
            shutil.copyfile(image_dir / file_name, out_dir / new_name)

            merged.append(image_meta)

    with open('./merged.json', 'w') as wf:
        json.dump({"prompts": prompts, "images": merged}, wf)

def to_hash(string):
    sha = hashlib.sha256()
    sha.update(string.encode())
    return sha.hexdigest()[:7]

def compare_index(i0, i1):
    # Sort by repetitions
    n0, d0 = i0["reps"].split("/")
    n1, d1 = i1["reps"].split("/")
    if (int(d0) != int(d1)):
        return int(d0) - int(d1)
    if (int(n0) != int(n1)):
        return int(n0) - int(n1)
    # Sort by version
    v0 = StrictVersion(i0["v"])
    v1 = StrictVersion(i1["v"])
    return v0 - v1

if __name__ == "__main__":
    main()
