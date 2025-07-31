import json
import os
import subprocess
import shutil
from algorithms.common import load_hdf5_image, save_temp_mat, delete_temp_mat 

with open("config.json") as f:
    cfg = json.load(f)

# load the file and save as temp_input.mat
image = load_hdf5_image(cfg["image_path"], slice_index=cfg.get("slice_index", 0))
save_temp_mat(image)

try:
    algo = cfg["algorithm"]
    algo_dir = os.path.join("algorithms", algo)
    runner_script = os.path.join(algo_dir, f"{algo}_runner.py")
    matlab_runner_script = os.path.join(algo_dir, f"{algo}_runner.m")

    with open(os.path.join(algo_dir, "env.json")) as f:
        env_cfg = json.load(f)
    env_name = env_cfg["env_name"].lower()

    project_root = os.path.abspath(os.path.dirname(__file__))

    # switch between matlab and other python environments
    if env_name.lower() == "matlab":
        default_matlab_path = "/Applications/MATLAB_R2025a.app/bin/maca64/MATLAB"  # my matlab path
        matlab_cmd = shutil.which("matlab") or default_matlab_path

        if not os.path.exists(matlab_cmd):
            raise RuntimeError("Can't find Matlab")

        runner_abs = os.path.join(project_root, matlab_runner_script).replace("\\", "/")
        project_root_escaped = project_root.replace("\\", "/")

        matlab_code = (
            f"try, cd('{project_root_escaped}'); run('{runner_abs}'); "
            f"catch ME, disp(getReport(ME)); end; exit"
        )

        subprocess.run(
            [matlab_cmd, "-nodesktop", "-nosplash", "-r", matlab_code],
            check=True,
            cwd=project_root
        )


    else:
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root

        runner_script_rel = os.path.join("algorithms", algo, f"{algo}_runner.py")

        subprocess.run(
            ["conda", "run", "-n", env_name, "python", runner_script_rel],
            check=True,
            cwd=project_root,
            env=env
        )

finally:
    # delete temporary file
    delete_temp_mat()
