from pathlib import Path
import subprocess
import json

'''
    This file consults the manifest to check if the single experiments
    are pending and, if so, runs it. After each experiment the manifest
    is updated with the outcome of the experiment.
'''

# CHANGE THE MANIFEST PATH IF NEEDED
MANIFEST_PATH = Path(r"../experiments/runs-preprocessing/manifest.json")


def read_manifest():
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
        return manifest

def write_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


manifest = read_manifest()

for run_id, run_parameters in manifest.items():
    if run_parameters["status"] == "PENDING":
        print(f"\nProcessing {run_id}:")
        # I derive the absolute path of the run folder
        absolute_path_run_folder = Path(run_parameters["run_folder"]).resolve()
        try:
            cmd = [
                "python",
                "single_run.py",
                "--absolute_path_run_folder", str(absolute_path_run_folder),
            ]
            subprocess.run(cmd, check=True)
            manifest[run_id]["status"] = "DONE"

        except subprocess.CalledProcessError as e:
            print(f"An error occurred during the execution of the experiment:\n{e}")
            manifest[run_id]["status"] = "FAILED"

        finally:
            # I update the manifest file
            write_manifest(manifest)
