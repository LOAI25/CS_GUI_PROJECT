import json
import os
import subprocess
import shutil
from algorithms.common import load_hdf5_image, save_temp_mat, delete_temp_mat

# === 读取配置 ===
with open("config.json") as f:
    cfg = json.load(f)

# === 加载 HDF5 图像（会自动取 slice_index 对应的切片，如果是 3D） ===
image = load_hdf5_image(cfg["image_path"], slice_index=cfg.get("slice_index", 0))

# === 如果有 ROI，则裁剪 ===
if "roi" in cfg:
    r = cfg["roi"]
    y_start, y_end = r["y_start"], r["y_end"]
    x_start, x_end = r["x_start"], r["x_end"]

    # 安全检查，防止越界
    H, W = image.shape
    y_start = max(0, min(y_start, H - 1))
    y_end = max(0, min(y_end, H - 1))
    x_start = max(0, min(x_start, W - 1))
    x_end = max(0, min(x_end, W - 1))

    image = image[y_start:y_end + 1, x_start:x_end + 1]
    print(f"ROI applied: x=({x_start}, {x_end}), y=({y_start}, {y_end}), shape={image.shape}")
else:
    print(f"No ROI, using full image, shape={image.shape}")

# === 保存临时输入文件 ===
save_temp_mat(image)

try:
    algo = cfg["algorithm"]
    algo_dir = os.path.join("algorithms", algo)
    runner_script = os.path.join(algo_dir, f"{algo}_runner.py")
    matlab_runner_script = os.path.join(algo_dir, f"{algo}_runner.m")

    # 读取运行环境
    with open(os.path.join(algo_dir, "env.json")) as f:
        env_cfg = json.load(f)
    env_name = env_cfg["env_name"].lower()

    project_root = os.path.abspath(os.path.dirname(__file__))
    print("=== Running algorithm:", algo, "===")

    # === MATLAB 环境 ===
    if env_name == "matlab":
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
            cwd=project_root
        )

    # === Python 环境 ===
    else:
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root

        runner_script_rel = os.path.join("algorithms", algo, f"{algo}_runner.py")

        if env_name.startswith("/"):  # 如果是绝对路径
            python_executable = os.path.join(env_name, "bin", "python")
            subprocess.run(
                [python_executable, runner_script_rel],
                check=True,
                cwd=project_root,
                env=env
            )
        else:  # 如果只是名字
            subprocess.run(
                ["conda", "run", "-n", env_name, "python", runner_script_rel],
                check=True,
                cwd=project_root,
                env=env
            )


finally:
    delete_temp_mat()
