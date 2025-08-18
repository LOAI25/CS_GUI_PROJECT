import sys
import json
import os
import subprocess
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import math
import glob
import time


from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QHBoxLayout,
    QDialog,
    QListWidget,
    QListWidgetItem,
)

from PyQt5.QtCore import QThread, pyqtSignal

from algorithms.common import (
    get_hdf5_image_dims,
    generate_sampling_mask,
    save_mask_to_mat,
    load_hdf5_image,
)


class AlgorithmRunner(QThread):
    """Run the specific compressed sensing algorithm in a background thread by calling run_algorithm.py, avoiding blocking the main thread"""

    finished = pyqtSignal(str)  # Success -- algorithm name
    failed = pyqtSignal(str, str)  # Fail -- alogorithm name + error meaasage

    def __init__(self, algo_name: str, parent=None):
        super().__init__(parent)
        self.algo_name = algo_name

    def run(self):
        try:
            subprocess.run(["python", "run_algorithm.py"], check=True)
            self.finished.emit(self.algo_name)
        except subprocess.CalledProcessError as e:
            self.failed.emit(self.algo_name, str(e))


class CS_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI for different kinds of compressed sensing")
        self.image_path = None
        self.image_shape = None
        self.advanced_params = {}
        self.initUI()

    def initUI(self):
        # Main interface
        self.layout = QVBoxLayout()

        # Choose algorithm
        self.alg_list_label = QLabel("Choose your algorithms (multi-select allowed):")
        self.alg_list = QListWidget()
        self.alg_list.setSelectionMode(self.alg_list.MultiSelection)
        for name in ["cvx", "BSBL_FM", "spg_bp", "CoSaMP", "FISTA"]:
            item = QListWidgetItem(name)
            self.alg_list.addItem(item)

        self.sampling_label = QLabel("Sampling rate:")
        self.sampling_input = QDoubleSpinBox()
        self.sampling_input.setRange(0.01, 1.0)
        self.sampling_input.setSingleStep(0.05)
        self.sampling_input.setValue(0.3)

        self.sampling_method_label = QLabel("Sampling method:")
        self.sampling_method_selector = QComboBox()
        self.sampling_method_selector.addItems(["random", "linehop"])

        self.use_clean_image_checkbox = QCheckBox("Use clean image (no noise)")
        self.use_clean_image_checkbox.setChecked(True)
        self.use_clean_image_checkbox.stateChanged.connect(self.toggle_snr_input)

        # Hide when checked
        self.snr_label = QLabel("SNR (dB):")
        self.snr_input = QSpinBox()
        self.snr_input.setRange(0, 100)
        self.snr_input.setValue(30)
        self.snr_label.setVisible(False)
        self.snr_input.setVisible(False)

        self.patch_label = QLabel("Patch size:")
        self.patch_size_input = QSpinBox()
        self.patch_size_input.setRange(8, 512)
        self.patch_size_input.setValue(32)

        self.stride_label = QLabel("Stride:")
        self.stride_input = QSpinBox()
        self.stride_input.setRange(1, 512)
        self.stride_input.setValue(16)

        self.slice_label = QLabel("Slice index:")
        self.slice_index_input = QSpinBox()
        self.slice_index_input.setRange(0, 999)
        self.slice_index_input.setValue(0)
        self.slice_label.setVisible(False)
        self.slice_index_input.setVisible(False)

        self.use_full_image_checkbox = QCheckBox("Use full image")
        self.use_full_image_checkbox.setChecked(True)
        self.use_full_image_checkbox.stateChanged.connect(self.toggle_roi_inputs)
        self.use_full_image_checkbox.setVisible(
            False
        )  # Initially hide the checkbox until an image is loaded

        self.roi_label = QLabel("ROI coordinates (x_start, x_end, y_start, y_end):")
        roi_layout = QHBoxLayout()
        self.x_start_input = QSpinBox()
        self.x_end_input = QSpinBox()
        self.y_start_input = QSpinBox()
        self.y_end_input = QSpinBox()
        roi_layout.addWidget(self.x_start_input)
        roi_layout.addWidget(self.x_end_input)
        roi_layout.addWidget(self.y_start_input)
        roi_layout.addWidget(self.y_end_input)

        # Hide when checked
        self.roi_label.setVisible(False)
        self.x_start_input.setVisible(False)
        self.x_end_input.setVisible(False)
        self.y_start_input.setVisible(False)
        self.y_end_input.setVisible(False)

        self.load_button = QPushButton("Choose your HDF5 file")
        self.load_button.clicked.connect(self.load_file)

        self.advanced_button = QPushButton("Advanced Settings")
        self.advanced_button.clicked.connect(self.open_advanced_settings)
        self.alg_list.itemSelectionChanged.connect(self.on_alg_selection_changed)
        self.on_alg_selection_changed()

        self.run_button = QPushButton("Do compressed sensing")
        self.run_button.clicked.connect(self.run_cs)

        # Initial status
        self.status_label = QLabel("The file hasn't been loaded")

        # Layout
        self.layout.addWidget(self.alg_list_label)
        self.layout.addWidget(self.alg_list)
        self.layout.addWidget(self.sampling_label)
        self.layout.addWidget(self.sampling_input)
        self.layout.addWidget(self.sampling_method_label)
        self.layout.addWidget(self.sampling_method_selector)
        self.layout.addWidget(self.use_clean_image_checkbox)
        self.layout.addWidget(self.snr_label)
        self.layout.addWidget(self.snr_input)
        self.layout.addWidget(self.patch_label)
        self.layout.addWidget(self.patch_size_input)
        self.layout.addWidget(self.stride_label)
        self.layout.addWidget(self.stride_input)
        self.layout.addWidget(self.slice_label)
        self.layout.addWidget(self.slice_index_input)
        self.layout.addWidget(self.use_full_image_checkbox)
        self.layout.addWidget(self.roi_label)
        self.layout.addLayout(roi_layout)
        self.layout.addWidget(self.advanced_button)
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.status_label)

        self.setLayout(self.layout)

    def on_alg_selection_changed(self):
        """Disable the advanced settings button only when all selected algorithms are spg_bp"""
        selected = [i.text() for i in self.alg_list.selectedItems()]
        only_spg = (len(selected) > 0) and all(a == "spg_bp" for a in selected)
        self.advanced_button.setEnabled(not only_spg)

    def toggle_snr_input(self):
        """Make the SNR input visible when unchecked"""
        visible = not self.use_clean_image_checkbox.isChecked()
        self.snr_label.setVisible(visible)
        self.snr_input.setVisible(visible)

    def toggle_roi_inputs(self):
        """Make the ROI input visible when unchecked"""
        visible = not self.use_full_image_checkbox.isChecked()
        self.roi_label.setVisible(visible)
        self.x_start_input.setVisible(visible)
        self.x_end_input.setVisible(visible)
        self.y_start_input.setVisible(visible)
        self.y_end_input.setVisible(visible)

    def open_advanced_settings(self):
        selected = [i.text() for i in self.alg_list.selectedItems()]
        if not selected:
            selected = [
                self.alg_list.item(0).text()
            ]  # If no algorithm is selected, use the first one as default

        # Open a new window
        dlg = QDialog(self)
        dlg.setWindowTitle("Advanced Settings")
        layout = QVBoxLayout()

        algo_selector = QComboBox()
        for a in selected:
            algo_selector.addItem(a)
        layout.addWidget(QLabel("Edit algorithm:"))
        layout.addWidget(algo_selector)

        form_container = QVBoxLayout()
        layout.addLayout(form_container)

        widgets_by_algo = {}

        def build_form_for(algo):
            # Clear old widgets
            while form_container.count():
                item = form_container.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()

            params = self.advanced_params.get(algo, {})

            if algo == "FISTA":
                lam_label = QLabel("lambda:")
                lam_input = QDoubleSpinBox()
                lam_input.setRange(1e-6, 1.0)
                lam_input.setDecimals(6)
                lam_input.setValue(params.get("lambda", 0.02))

                bt_label = QLabel("Use backtracking:")
                bt_checkbox = QCheckBox()
                bt_checkbox.setChecked(params.get("fista_backtracking", False))

                L0_label = QLabel("L0:")
                L0_input = QDoubleSpinBox()
                L0_input.setRange(1e-6, 1e6)
                L0_input.setDecimals(6)
                L0_input.setValue(params.get("L0", 1.0))

                eta_label = QLabel("eta:")
                eta_input = QDoubleSpinBox()
                eta_input.setRange(1.0, 10.0)
                eta_input.setDecimals(3)
                eta_input.setValue(params.get("eta", 1.5))

                it_label = QLabel("Max Iters:")
                it_input = QSpinBox()
                it_input.setRange(10, 100000)
                it_input.setValue(params.get("max_iters", 500))

                eps_label = QLabel("Epsilon:")
                eps_input = QDoubleSpinBox()
                eps_input.setRange(1e-12, 1e-2)
                eps_input.setDecimals(12)
                eps_input.setSingleStep(1e-6)
                eps_input.setValue(params.get("epsilon", 1e-6))

                pos_label = QLabel("Nonnegative coefficients:")
                pos_checkbox = QCheckBox()
                pos_checkbox.setChecked(params.get("fista_pos", False))

                form_container.addWidget(lam_label)
                form_container.addWidget(lam_input)
                form_container.addWidget(bt_label)
                form_container.addWidget(bt_checkbox)
                form_container.addWidget(L0_label)
                form_container.addWidget(L0_input)
                form_container.addWidget(eta_label)
                form_container.addWidget(eta_input)
                form_container.addWidget(it_label)
                form_container.addWidget(it_input)
                form_container.addWidget(eps_label)
                form_container.addWidget(eps_input)
                form_container.addWidget(pos_label)
                form_container.addWidget(pos_checkbox)

                widgets_by_algo[algo] = {
                    "lambda": lam_input,
                    "fista_backtracking": bt_checkbox,
                    "L0": L0_input,
                    "eta": eta_input,
                    "max_iters": it_input,
                    "epsilon": eps_input,
                    "fista_pos": pos_checkbox,
                }

            elif algo == "cvx":
                lam_label = QLabel("Lambda:")
                lam_input = QDoubleSpinBox()
                lam_input.setRange(0.0001, 1.0)
                lam_input.setDecimals(4)
                lam_input.setValue(params.get("lam", 0.01))

                form_container.addWidget(lam_label)
                form_container.addWidget(lam_input)

                widgets_by_algo[algo] = {"lam": lam_input}

            elif algo == "CoSaMP":
                maxiter_label = QLabel("Max Iters:")
                maxiter_input = QSpinBox()
                maxiter_input.setRange(1, 1000)
                maxiter_input.setValue(params.get("max_iters", 200))

                k_label = QLabel("Sparsity K:")
                k_input = QSpinBox()
                k_input.setRange(1, 4096)
                k_input.setValue(params.get("K", 30))

                form_container.addWidget(maxiter_label)
                form_container.addWidget(maxiter_input)
                form_container.addWidget(k_label)
                form_container.addWidget(k_input)

                widgets_by_algo[algo] = {"max_iters": maxiter_input, "K": k_input}

            elif algo == "BSBL_FM":
                blklen_label = QLabel("Blk_len:")
                blklen_input = QSpinBox()
                blklen_input.setRange(1, 64)
                blklen_input.setValue(params.get("blk_len", 1))

                lambda_label = QLabel("Learn Lambda:")
                lambda_selector = QComboBox()
                lambda_selector.addItems(["No learning rule", "Medium SNR", "High SNR"])
                lambda_selector.setCurrentIndex(params.get("learn_lambda", 0))

                maxiter_label = QLabel("Max Iters:")
                maxiter_input = QSpinBox()
                maxiter_input.setRange(10, 10000)
                maxiter_input.setValue(params.get("max_iters", 500))

                learntype_label = QLabel("Learn Type:")
                learntype_selector = QComboBox()
                learntype_selector.addItems(
                    [
                        "No intra-correlation",
                        "Individual intra-correlation",
                        "Unified intra-correlation",
                    ]
                )
                learntype_selector.setCurrentIndex(params.get("learntype", 0))

                epsilon_label = QLabel("Epsilon:")
                epsilon_input = QDoubleSpinBox()
                epsilon_input.setDecimals(8)
                epsilon_input.setRange(1e-10, 1e-2)
                epsilon_input.setSingleStep(1e-7)
                epsilon_input.setValue(params.get("epsilon", 1e-7))

                form_container.addWidget(blklen_label)
                form_container.addWidget(blklen_input)
                form_container.addWidget(lambda_label)
                form_container.addWidget(lambda_selector)
                form_container.addWidget(maxiter_label)
                form_container.addWidget(maxiter_input)
                form_container.addWidget(learntype_label)
                form_container.addWidget(learntype_selector)
                form_container.addWidget(epsilon_label)
                form_container.addWidget(epsilon_input)

                widgets_by_algo[algo] = {
                    "blk_len": blklen_input,
                    "learn_lambda": lambda_selector,
                    "max_iters": maxiter_input,
                    "learntype": learntype_selector,
                    "epsilon": epsilon_input,
                }

            else:
                note = QLabel(f"{algo} has no advanced parameters.")
                form_container.addWidget(note)
                widgets_by_algo[algo] = {}

        build_form_for(algo_selector.currentText())
        algo_selector.currentTextChanged.connect(build_form_for)

        ok_btn = QPushButton("OK")

        def save_and_close():
            algo = algo_selector.currentText()
            w = widgets_by_algo.get(algo, {})
            self.advanced_params.setdefault(algo, {})

            if algo == "FISTA":
                self.advanced_params[algo]["lambda"] = w["lambda"].value()
                self.advanced_params[algo]["fista_backtracking"] = w[
                    "fista_backtracking"
                ].isChecked()
                self.advanced_params[algo]["L0"] = w["L0"].value()
                self.advanced_params[algo]["eta"] = w["eta"].value()
                self.advanced_params[algo]["max_iters"] = w["max_iters"].value()
                self.advanced_params[algo]["epsilon"] = w["epsilon"].value()
                self.advanced_params[algo]["fista_pos"] = w["fista_pos"].isChecked()

            elif algo == "cvx":
                self.advanced_params[algo]["lam"] = w["lam"].value()

            elif algo == "CoSaMP":
                self.advanced_params[algo]["max_iters"] = w["max_iters"].value()
                self.advanced_params[algo]["K"] = w["K"].value()

            elif algo == "BSBL_FM":
                self.advanced_params[algo]["blk_len"] = w["blk_len"].value()
                self.advanced_params[algo]["learn_lambda"] = w[
                    "learn_lambda"
                ].currentIndex()
                self.advanced_params[algo]["max_iters"] = w["max_iters"].value()
                self.advanced_params[algo]["learntype"] = w["learntype"].currentIndex()
                self.advanced_params[algo]["epsilon"] = w["epsilon"].value()

            dlg.accept()

        ok_btn.clicked.connect(save_and_close)
        layout.addWidget(ok_btn)
        dlg.setLayout(layout)
        dlg.exec_()

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choose your HDF5 file", "", "HDF5 Files (*.hdf5)"
        )  # Restrict to loading only HDF5 files
        MIN_ROI_SIZE = 8
        if file_path:
            self.image_path = file_path
            try:
                shape = get_hdf5_image_dims(file_path)
                self.image_shape = shape
                ndim = len(shape)

                if ndim == 2:
                    H, W = shape
                    self.slice_label.setVisible(False)
                    self.slice_index_input.setVisible(False)
                elif ndim == 3:
                    H, W = shape[1], shape[2]
                    self.slice_label.setVisible(True)
                    self.slice_index_input.setVisible(True)
                    self.slice_index_input.setMaximum(shape[0] - 1)
                else:
                    self.status_label.setText(f"Wrong image dim: {shape}")
                    return

                # Check the minimal size
                if H < MIN_ROI_SIZE or W < MIN_ROI_SIZE:
                    self.status_label.setText(
                        f"Image size too small (must be at least {MIN_ROI_SIZE} pixels in both dimensions)"
                    )
                    return

                # Set the range of ROI
                self.x_start_input.setRange(0, W - MIN_ROI_SIZE)
                self.x_end_input.setRange(MIN_ROI_SIZE - 1, W - 1)
                self.y_start_input.setRange(0, H - MIN_ROI_SIZE)
                self.y_end_input.setRange(MIN_ROI_SIZE - 1, H - 1)

                self.x_start_input.setValue(0)
                self.y_start_input.setValue(0)
                self.x_end_input.setValue(min(63, W - 1))
                self.y_end_input.setValue(min(63, H - 1))

                # Dynamically adjust with a minimum size no less than 8
                def update_x_end_limit():
                    start = self.x_start_input.value()
                    self.x_end_input.setMinimum(start + MIN_ROI_SIZE - 1)

                def update_x_start_limit():
                    end = self.x_end_input.value()
                    self.x_start_input.setMaximum(end - MIN_ROI_SIZE + 1)

                def update_y_end_limit():
                    start = self.y_start_input.value()
                    self.y_end_input.setMinimum(start + MIN_ROI_SIZE - 1)

                def update_y_start_limit():
                    end = self.y_end_input.value()
                    self.y_start_input.setMaximum(end - MIN_ROI_SIZE + 1)

                self.x_start_input.valueChanged.connect(update_x_end_limit)
                self.x_end_input.valueChanged.connect(update_x_start_limit)
                self.y_start_input.valueChanged.connect(update_y_end_limit)
                self.y_end_input.valueChanged.connect(update_y_start_limit)

                self.status_label.setText(f"File loaded with dim: {shape}")
                self.use_full_image_checkbox.setVisible(True)

            except Exception as e:
                self.status_label.setText(f"Failed to load file: {e}")

    def run_cs(self):
        if not self.image_path:
            self.status_label.setText("The file hasn't been selected")
            return
        if self.image_shape is None:
            self.status_label.setText("Image not properly loaded.")
            return

        algos = [i.text() for i in self.alg_list.selectedItems()]
        if not algos:
            self.status_label.setText("Please select at least one algorithm.")
            return

        cfg_base = {
            "image_path": self.image_path,
            "slice_index": self.slice_index_input.value(),
            "sampling_rate": self.sampling_input.value(),
            "patch_size": self.patch_size_input.value(),
            "stride": self.stride_input.value(),
        }

        if not self.use_full_image_checkbox.isChecked():
            cfg_base["roi"] = {
                "x_start": self.x_start_input.value(),
                "x_end": self.x_end_input.value(),
                "y_start": self.y_start_input.value(),
                "y_end": self.y_end_input.value(),
            }

        if not self.use_clean_image_checkbox.isChecked():
            cfg_base["snr"] = self.snr_input.value()

        # Check the image size. If ROI is used, then use the size of the sub-image
        if "roi" in cfg_base:
            H = cfg_base["roi"]["y_end"] - cfg_base["roi"]["y_start"] + 1
            W = cfg_base["roi"]["x_end"] - cfg_base["roi"]["x_start"] + 1
        else:
            if len(self.image_shape) == 2:
                H, W = self.image_shape
            else:
                H, W = self.image_shape[1], self.image_shape[2]

        try:
            # Generate sampling mask and save as a temporary file
            mask = generate_sampling_mask(
                H,
                W,
                self.sampling_input.value(),
                method=self.sampling_method_selector.currentText(),
                seed=42,
            )
            save_mask_to_mat(mask, "sampling_mask.mat")
        except Exception as e:
            self.status_label.setText(f"Fail to generate sampling mask: {e}")
            return

        try:
            # The noise addition and observed_display below are ONLY for GUI preview.
            # They are NOT saved into temp_input.mat and will NOT be passed to run_algorithm.py or any runner.

            # Load the original image according to ROI and slice
            full_img = load_hdf5_image(
                cfg_base["image_path"], slice_index=cfg_base.get("slice_index", 0)
            )
            if "roi" in cfg_base:
                r = cfg_base["roi"]
                full_img = full_img[
                    r["y_start"] : r["y_end"] + 1, r["x_start"] : r["x_end"] + 1
                ]
            # Add noise
            if "snr" in cfg_base:
                sigma = np.std(full_img) * (10 ** (-cfg_base["snr"] / 20))
                noisy = full_img + sigma * np.random.randn(*full_img.shape)
                noisy = np.clip(noisy, 0, 1)
            else:
                noisy = full_img.copy()
            # Make an observed image
            observed_display = noisy * mask.astype(noisy.dtype)
            # For show
            self._display_full_img = full_img
            self._display_observed_img = observed_display
        except Exception as e:
            self.status_label.setText(f"Fail to prepare display images: {e}")
            return

        # Generate a configuration dictionary for each algorithm, prepared for config.json
        self._algo_queue = []
        for algo in algos:
            cfg = dict(cfg_base)  # Copy common parameters such as slice, ROI, SNR, etc.
            cfg["algorithm"] = algo
            cfg["output_path"] = f"reconstructed_{algo}.mat"
            cfg["metrics_path"] = f"metrics_{algo}.json"

            # Add advanced parameters
            adv = self.advanced_params.get(algo, {})
            if algo == "BSBL_FM":
                cfg["blk_len"] = adv.get("blk_len", 1)
                cfg["learn_lambda"] = adv.get("learn_lambda", 0)
                cfg["max_iters"] = adv.get("max_iters", 500)
                cfg["epsilon"] = adv.get("epsilon", 1e-7)
                cfg["learntype"] = adv.get("learntype", 0)
            elif algo == "cvx":
                cfg["lam"] = adv.get("lam", 0.01)
            elif algo == "CoSaMP":
                cfg["max_iters"] = adv.get("max_iters", 200)
                cfg["K"] = adv.get("K", 30)
            elif algo == "FISTA":
                cfg["lambda"] = adv.get("lambda", 0.02)
                cfg["fista_backtracking"] = adv.get("fista_backtracking", False)
                cfg["L0"] = adv.get("L0", 1.0)
                cfg["eta"] = adv.get("eta", 1.5)
                cfg["max_iters"] = adv.get("max_iters", 500)
                cfg["epsilon"] = adv.get("epsilon", 1e-6)
                cfg["fista_pos"] = adv.get("fista_pos", False)
            elif algo == "spg_bp":
                pass

            self._algo_queue.append((algo, cfg))

        self._results = {}

        def start_next():
            # If the queue is empty which means there's no algorithm need to be run, show the final result
            if not self._algo_queue:
                self.show_comparison_figure()
                return
            # Use the next algorithm and delete it from the queue
            algo, cfg = self._algo_queue.pop(0)
            # Write config.json
            with open("config.json", "w") as f:
                json.dump(cfg, f)
            # Update the status bar, let users know which algorithm is running
            self.status_label.setText(f"Running {algo} ...")
            QApplication.processEvents()

            # Record the start time
            if not hasattr(self, "_algo_t0"):
                self._algo_t0 = {}
            self._algo_t0[algo] = time.perf_counter()

            self.runner = AlgorithmRunner(algo)
            self.runner.finished.connect(on_finished)
            self.runner.failed.connect(on_failed)
            self.runner.start()

        def on_finished(algo_name):
            try:
                out_mat = f"reconstructed_{algo_name}.mat"
                if os.path.exists(out_mat):
                    mat_data = loadmat(out_mat)
                    recon = mat_data["recon_img"]
                else:
                    recon = None

                metrics_path = f"metrics_{algo_name}.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    psnr_val = metrics.get("psnr", 0.0)
                    ssim_val = metrics.get("ssim", 0.0)
                else:
                    psnr_val = ssim_val = 0.0

                # Stop timing and calculate elapsed time
                t0 = getattr(self, "_algo_t0", {}).pop(algo_name, None)
                elapsed = (time.perf_counter() - t0) if t0 is not None else 0.0

                self._results[algo_name] = {
                    "recon": recon,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "time": elapsed,
                }

                # Write the elapsed time back to the metrics file of this algorithm
                try:
                    if os.path.exists(metrics_path):
                        metrics["time_sec"] = elapsed
                        with open(metrics_path, "w") as f:
                            json.dump(metrics, f)
                except Exception:
                    pass

            except Exception as e:
                self.status_label.setText(
                    f"{algo_name} finished but failed to load outputs: {e}"
                )
            start_next()

        def on_failed(algo_name, error_msg):
            # If failed, remove this algorithm's record from the timer
            if hasattr(self, "_algo_t0"):
                self._algo_t0.pop(algo_name, None)

            self.status_label.setText(f"Fail to run {algo_name}: {error_msg}")
            start_next()

        self.status_label.setText("The algorithms are working (queued)")
        QApplication.processEvents()
        start_next()

    def show_comparison_figure(self):
        tiles = []
        titles = []
        subtitles = []

        tiles.append(self._display_full_img)
        titles.append("Original")
        subtitles.append("")

        tiles.append(self._display_observed_img)
        titles.append(
            "Observed (sampled{})".format(
                ""
                if self.use_clean_image_checkbox.isChecked()
                else f", SNR={self.snr_input.value()} dB"
            )
        )
        subtitles.append("")

        for algo, res in self._results.items():
            img = res.get("recon", None)
            if img is None:
                continue
            tiles.append(img)
            titles.append(algo)
            subtitles.append(
                f"PSNR={res.get('psnr',0):.2f} dB, SSIM={res.get('ssim',0):.4f}, t={res.get('time',0):.2f}s"
            )

        n = len(tiles)
        if n == 0:
            self.status_label.setText("No images to display.")
            return

        # Arrange images in an adaptive grid layout
        cols = 3 if n <= 6 else 4
        rows = math.ceil(n / cols)

        plt.figure(figsize=(4 * cols, 3.8 * rows))
        for idx, (img, t, sub) in enumerate(zip(tiles, titles, subtitles), start=1):
            ax = plt.subplot(rows, cols, idx)
            ax.imshow(img, cmap="gray")
            ax.set_title(t, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            if sub:
                ax.set_xlabel(sub, fontsize=9)
        plt.tight_layout()
        plt.show()
        if self._results:
            last_algo = list(self._results.keys())[-1]
            m = self._results[last_algo]
            self.status_label.setText(
                f"Last: {last_algo}  PSNR={m.get('psnr',0):.2f}, SSIM={m.get('ssim',0):.4f}"
            )
        else:
            self.status_label.setText("Done (no recon images)")

    def closeEvent(self, event):
        # Wait for possible running subtasks to finish, to avoid leftover temporary files
        if hasattr(self, "runner") and self.runner and self.runner.isRunning():
            self.runner.wait(3000)
        for p in ["sampling_mask.mat", "temp_input.mat", "config.json"]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        for pat in ("reconstructed_*.mat", "metrics_*.json"):
            for p in glob.glob(pat):
                try:
                    os.remove(p)
                except Exception:
                    pass

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CS_GUI()
    gui.show()
    sys.exit(app.exec_())
