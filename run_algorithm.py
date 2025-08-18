import json
import os
import subprocess
import shutil
from algorithms.common import load_hdf5_image, save_temp_mat, delete_temp_mat

# Read config.json
with open("config.json") as f:
    cfg = json.load(f)

# Load HDF5 file
image = load_hdf5_image(cfg["image_path"], slice_index=cfg.get("slice_index", 0))

if "roi" in cfg:
    r = cfg["roi"]
    y_start, y_end = r["y_start"], r["y_end"]
    x_start, x_end = r["x_start"], r["x_end"]

    # Prevent out-of-bounds by clamping ROI coordinates within image size
    H, W = image.shape
    y_start = max(0, min(y_start, H - 1))
    y_end = max(0, min(y_end, H - 1))
    x_start = max(0, min(x_start, W - 1))
    x_end = max(0, min(x_end, W - 1))

    image = image[y_start : y_end + 1, x_start : x_end + 1]
    print(
        f"ROI applied: x=({x_start}, {x_end}), y=({y_start}, {y_end}), shape={image.shape}"
    )
else:
    print(f"No ROI, using full image, shape={image.shape}")

# Save image as a temporary file
save_temp_mat(image)

# Execute depending on the environment
try:
    algo = cfg["algorithm"]
    algo_dir = os.path.join("algorithms", algo)
    runner_script = os.path.join(algo_dir, f"{algo}_runner.py")
    matlab_runner_script = os.path.join(algo_dir, f"{algo}_runner.m")

    # Read env.json
    with open(os.path.join(algo_dir, "env.json")) as f:
        env_cfg = json.load(f)
    env_name = env_cfg["env_name"].lower()

    project_root = os.path.abspath(os.path.dirname(__file__))
    print("Running algorithm:", algo)

    # MATLAB
    if env_name == "matlab":
        # Set the default MATLAB path. This is the path on my computer, please change it to the MATLAB path on your computer.
        default_matlab_path = "/Applications/MATLAB_R2025a.app/bin/maca64/MATLAB"
        matlab_cmd = shutil.which("matlab") or default_matlab_path

        if not os.path.exists(matlab_cmd):
            raise RuntimeError("Can't find Matlab executable.")

        runner_abs = os.path.join(project_root, matlab_runner_script).replace("\\", "/")
        project_root_escaped = project_root.replace("\\", "/")

        matlab_code = (
            f"try, cd('{project_root_escaped}'); run('{runner_abs}'); "
            f"catch ME, disp(getReport(ME)); end; exit"
        )

        subprocess.run(
            [matlab_cmd, "-nodesktop", "-nosplash", "-r", matlab_code],
            check=True,
            cwd=project_root,
        )

    # Python
    else:
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root

        runner_script_rel = os.path.join("algorithms", algo, f"{algo}_runner.py")

        if env_name.startswith(
            "/"
        ):  # If env_name is an absolute path (specifying the virtual environment path)
            python_executable = os.path.join(env_name, "bin", "python")
            subprocess.run(
                [python_executable, runner_script_rel],
                check=True,
                cwd=project_root,
                env=env,
            )
        else:  # Just a name -- conda environment
            subprocess.run(
                ["conda", "run", "-n", env_name, "python", runner_script_rel],
                check=True,
                cwd=project_root,
                env=env,
            )


finally:
    delete_temp_mat()
