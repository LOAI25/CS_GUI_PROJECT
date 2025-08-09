import sys
import json
import os
import subprocess
import matplotlib.pyplot as plt
from scipy.io import loadmat

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QHBoxLayout, QDialog
)
from PyQt5.QtCore import QThread, pyqtSignal

from algorithms.common import get_hdf5_image_dims, generate_sampling_mask, save_mask_to_mat


class AlgorithmRunner(QThread):
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self):
        try:
            subprocess.run(["python", "run_algorithm.py"], check=True)
            self.finished.emit()
        except subprocess.CalledProcessError as e:
            self.failed.emit(str(e))


class CS_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI for different kinds of compressed sensing")
        self.image_path = None
        self.image_shape = None
        self.roi = None
        self.advanced_params = {}  # 存放高级参数
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        # 算法选择
        self.alg_selector = QComboBox()
        self.alg_selector.addItems(["cvx", "BSBL_FM", "spg_bp", "CoSaMP", "FISTA"])


        # 采样率
        self.sampling_label = QLabel("Sampling rate:")
        self.sampling_input = QDoubleSpinBox()
        self.sampling_input.setRange(0.01, 1.0)
        self.sampling_input.setSingleStep(0.05)
        self.sampling_input.setValue(0.3)

        # 采样方法
        self.sampling_method_label = QLabel("Sampling method:")
        self.sampling_method_selector = QComboBox()
        self.sampling_method_selector.addItems(["random", "linehop"])

        # 是否添加噪声
        self.use_clean_image_checkbox = QCheckBox("Use clean image (no noise)")
        self.use_clean_image_checkbox.setChecked(True)
        self.use_clean_image_checkbox.stateChanged.connect(self.toggle_snr_input)

        self.snr_label = QLabel("SNR (dB):")
        self.snr_input = QSpinBox()
        self.snr_input.setRange(0, 100)
        self.snr_input.setValue(30)
        self.snr_label.setVisible(False)
        self.snr_input.setVisible(False)

        # Patch size
        self.patch_label = QLabel("Patch size:")
        self.patch_size_input = QSpinBox()
        self.patch_size_input.setRange(8, 512)
        self.patch_size_input.setValue(32)

        # Stride
        self.stride_label = QLabel("Stride:")
        self.stride_input = QSpinBox()
        self.stride_input.setRange(1, 512)
        self.stride_input.setValue(16)

        # Slice index
        self.slice_label = QLabel("Slice index:")
        self.slice_index_input = QSpinBox()
        self.slice_index_input.setRange(0, 999)
        self.slice_index_input.setValue(0)
        self.slice_label.setVisible(False)
        self.slice_index_input.setVisible(False)

        # ROI
        self.use_full_image_checkbox = QCheckBox("Use full image")
        self.use_full_image_checkbox.setChecked(True)
        self.use_full_image_checkbox.stateChanged.connect(self.toggle_roi_inputs)

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
        self.roi_label.setVisible(False)
        self.x_start_input.setVisible(False)
        self.x_end_input.setVisible(False)
        self.y_start_input.setVisible(False)
        self.y_end_input.setVisible(False)

        # 按钮
        self.load_button = QPushButton("Choose your HDF5 file")
        self.load_button.clicked.connect(self.load_file)

        self.advanced_button = QPushButton("Advanced Settings")
        self.advanced_button.clicked.connect(self.open_advanced_settings)
        self.alg_selector.currentTextChanged.connect(self.toggle_advanced_button_visibility)


        self.run_button = QPushButton("Do compressed sensing")
        self.run_button.clicked.connect(self.run_cs)

        # 状态
        self.status_label = QLabel("The file hasn't been loaded")

        # 布局
        self.layout.addWidget(QLabel("Choose your algorithm:"))
        self.layout.addWidget(self.alg_selector)
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

    def toggle_advanced_button_visibility(self, text):
        if text == "spg_bp":
            self.advanced_button.setVisible(False)
        else:
            self.advanced_button.setVisible(True)

    def toggle_snr_input(self):
        visible = not self.use_clean_image_checkbox.isChecked()
        self.snr_label.setVisible(visible)
        self.snr_input.setVisible(visible)

    def toggle_roi_inputs(self):
        visible = not self.use_full_image_checkbox.isChecked()
        self.roi_label.setVisible(visible)
        self.x_start_input.setVisible(visible)
        self.x_end_input.setVisible(visible)
        self.y_start_input.setVisible(visible)
        self.y_end_input.setVisible(visible)

    def open_advanced_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Advanced Settings")
        layout = QVBoxLayout()

        algo = self.alg_selector.currentText()

        if algo == "BSBL_FM":
            blklen_label = QLabel("Blk_len:")
            blklen_input = QSpinBox()
            blklen_input.setRange(1, 64)
            blklen_input.setValue(self.advanced_params.get("blk_len", 1))
            layout.addWidget(blklen_label)
            layout.addWidget(blklen_input)

            lambda_label = QLabel("Learn Lambda:")
            lambda_selector = QComboBox()
            lambda_selector.addItems(["No learning rule", "Medium SNR", "High SNR"])
            lambda_selector.setCurrentIndex(self.advanced_params.get("learn_lambda", 0))
            layout.addWidget(lambda_label)
            layout.addWidget(lambda_selector)

            maxiter_label = QLabel("Max Iters:")
            maxiter_input = QSpinBox()
            maxiter_input.setRange(10, 10000)
            maxiter_input.setValue(self.advanced_params.get("max_iters", 500))
            layout.addWidget(maxiter_label)
            layout.addWidget(maxiter_input)

            learntype_label = QLabel("Learn Type:")
            learntype_selector = QComboBox()
            learntype_selector.addItems(["No intra-correlation", "Individual intra-correlation", "Unified intra-correlation"])
            learntype_selector.setCurrentIndex(self.advanced_params.get("learntype", 0))
            layout.addWidget(learntype_label)
            layout.addWidget(learntype_selector)

            epsilon_label = QLabel("Epsilon:")
            epsilon_input = QDoubleSpinBox()
            epsilon_input.setDecimals(8)
            epsilon_input.setRange(1e-10, 1e-2)
            epsilon_input.setSingleStep(1e-7)
            epsilon_input.setValue(self.advanced_params.get("epsilon", 1e-7))
            layout.addWidget(epsilon_label)
            layout.addWidget(epsilon_input)

        elif algo == "cvx":
            lam_label = QLabel("Lambda:")
            lam_input = QDoubleSpinBox()
            lam_input.setRange(0.0001, 1.0)
            lam_input.setDecimals(4)
            lam_input.setValue(self.advanced_params.get("lam", 0.01))
            layout.addWidget(lam_label)
            layout.addWidget(lam_input)

        elif algo == "CoSaMP":
            maxiter_label = QLabel("Max Iters:")
            maxiter_input = QSpinBox()
            maxiter_input.setRange(1, 1000)
            maxiter_input.setValue(self.advanced_params.get("max_iters", 200))
            layout.addWidget(maxiter_label)
            layout.addWidget(maxiter_input)

            k_label = QLabel("Sparsity K:")
            k_input = QSpinBox()
            k_input.setRange(1, 4096)
            k_input.setValue(self.advanced_params.get("K", 30))
            layout.addWidget(k_label)
            layout.addWidget(k_input)

        elif algo == "FISTA":
            lam_label = QLabel("lambda:")
            lam_input = QDoubleSpinBox()
            lam_input.setRange(1e-6, 1.0)
            lam_input.setDecimals(6)
            lam_input.setValue(self.advanced_params.get("lambda", 0.02))
            layout.addWidget(lam_label); layout.addWidget(lam_input)

            bt_label = QLabel("Use backtracking:")
            bt_checkbox = QCheckBox()
            bt_checkbox.setChecked(self.advanced_params.get("fista_backtracking", False))
            layout.addWidget(bt_label); layout.addWidget(bt_checkbox)

            L0_label = QLabel("L0:")
            L0_input = QDoubleSpinBox()
            L0_input.setRange(1e-6, 1e6); L0_input.setDecimals(6)
            L0_input.setValue(self.advanced_params.get("L0", 1.0))
            layout.addWidget(L0_label); layout.addWidget(L0_input)

            eta_label = QLabel("eta:")
            eta_input = QDoubleSpinBox()
            eta_input.setRange(1.0, 10.0); eta_input.setDecimals(3)
            eta_input.setValue(self.advanced_params.get("eta", 1.5))
            layout.addWidget(eta_label); layout.addWidget(eta_input)

            it_label = QLabel("Max Iters:")
            it_input = QSpinBox()
            it_input.setRange(10, 100000)
            it_input.setValue(self.advanced_params.get("max_iters", 500))
            layout.addWidget(it_label); layout.addWidget(it_input)

            eps_label = QLabel("Epsilon:")
            eps_input = QDoubleSpinBox()
            eps_input.setRange(1e-12, 1e-2); eps_input.setDecimals(12)
            eps_input.setSingleStep(1e-6)
            eps_input.setValue(self.advanced_params.get("epsilon", 1e-6))
            layout.addWidget(eps_label); layout.addWidget(eps_input)

            pos_label = QLabel("Nonnegative coefficients:")
            pos_checkbox = QCheckBox()
            pos_checkbox.setChecked(self.advanced_params.get("fista_pos", False))
            layout.addWidget(pos_label); layout.addWidget(pos_checkbox)


        ok_btn = QPushButton("OK")

        def save_and_close():
            if algo == "BSBL_FM":
                self.advanced_params["blk_len"] = blklen_input.value()
                self.advanced_params["learn_lambda"] = lambda_selector.currentIndex()
                self.advanced_params["max_iters"] = maxiter_input.value()
                self.advanced_params["epsilon"] = epsilon_input.value()
                self.advanced_params["learntype"] = learntype_selector.currentIndex()
            elif algo == "cvx":
                self.advanced_params["lam"] = lam_input.value()
            elif algo == "CoSaMP":
                self.advanced_params["max_iters"] = maxiter_input.value()
                self.advanced_params["K"] = k_input.value()
            elif algo == "FISTA":
                self.advanced_params["lambda"] = lam_input.value()
                self.advanced_params["fista_backtracking"] = bt_checkbox.isChecked()
                self.advanced_params["L0"] = L0_input.value()
                self.advanced_params["eta"] = eta_input.value()
                self.advanced_params["max_iters"] = it_input.value()
                self.advanced_params["epsilon"] = eps_input.value()
                self.advanced_params["fista_pos"] = pos_checkbox.isChecked()


            dlg.accept()

        ok_btn.clicked.connect(save_and_close)
        layout.addWidget(ok_btn)
        dlg.setLayout(layout)
        dlg.exec_()

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose your HDF5 file", "", "HDF5 Files (*.hdf5)")
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

                self.x_start_input.setRange(0, W - 1)
                self.x_end_input.setRange(0, W - 1)
                self.y_start_input.setRange(0, H - 1)
                self.y_end_input.setRange(0, H - 1)
                self.x_start_input.setValue(0)
                self.y_start_input.setValue(0)
                self.x_end_input.setValue(W - 1)
                self.y_end_input.setValue(H - 1)

                self.status_label.setText(f"File loaded with dim: {shape}")
            except Exception as e:
                self.status_label.setText(f"Failed to load file: {e}")

    def run_cs(self):
        if not self.image_path:
            self.status_label.setText("The file hasn't been selected")
            return
        if self.image_shape is None:
            self.status_label.setText("Image not properly loaded.")
            return

        algorithm = self.alg_selector.currentText()
        cfg = {
            "image_path": self.image_path,
            "slice_index": self.slice_index_input.value(),
            "sampling_rate": self.sampling_input.value(),
            "patch_size": self.patch_size_input.value(),
            "stride": self.stride_input.value(),
            "algorithm": algorithm,
            "output_path": "reconstructed.mat",
            "metrics_path": "metrics.json"
        }

        if not self.use_full_image_checkbox.isChecked():
            cfg["roi"] = {
                "x_start": self.x_start_input.value(),
                "x_end": self.x_end_input.value(),
                "y_start": self.y_start_input.value(),
                "y_end": self.y_end_input.value()
            }

        if not self.use_clean_image_checkbox.isChecked():
            cfg["snr"] = self.snr_input.value()

        if algorithm == "BSBL_FM":
            cfg["blk_len"] = self.advanced_params.get("blk_len", 1)
            cfg["learn_lambda"] = self.advanced_params.get("learn_lambda", 0)
            cfg["max_iters"] = self.advanced_params.get("max_iters", 500)
            cfg["epsilon"] = self.advanced_params.get("epsilon", 1e-7)
            cfg["learntype"] = self.advanced_params.get("learntype", 0)

        elif algorithm == "cvx":
            cfg["lam"] = self.advanced_params.get("lam", 0.01)
        elif algorithm == "CoSaMP":
            cfg["max_iters"] = self.advanced_params.get("max_iters", 200)
            cfg["K"] = self.advanced_params.get("K", 30)
        elif algorithm == "FISTA":
            cfg["lambda"] = self.advanced_params.get("lambda", 0.02)
            cfg["fista_backtracking"] = self.advanced_params.get("fista_backtracking", False)
            cfg["L0"] = self.advanced_params.get("L0", 1.0)
            cfg["eta"] = self.advanced_params.get("eta", 1.5)
            cfg["max_iters"] = self.advanced_params.get("max_iters", 500)
            cfg["epsilon"] = self.advanced_params.get("epsilon", 1e-6)
            cfg["fista_pos"] = self.advanced_params.get("fista_pos", False)



        elif algorithm == "spg_bp":
            pass  # 当前不需要额外参数


        with open("config.json", "w") as f:
            json.dump(cfg, f)

        try:
            if "roi" in cfg:
                H = cfg["roi"]["y_end"] - cfg["roi"]["y_start"] + 1
                W = cfg["roi"]["x_end"] - cfg["roi"]["x_start"] + 1
            else:
                if len(self.image_shape) == 2:
                    H, W = self.image_shape
                else:
                    H, W = self.image_shape[1], self.image_shape[2]

            mask = generate_sampling_mask(H, W, self.sampling_input.value(),
                                          method=self.sampling_method_selector.currentText())
            save_mask_to_mat(mask, "sampling_mask.mat")
        except Exception as e:
            self.status_label.setText(f"Fail to generate sampling mask: {e}")
            return

        self.status_label.setText("The algorithm is working")
        QApplication.processEvents()

        self.runner = AlgorithmRunner()
        self.runner.finished.connect(self.on_run_finished)
        self.runner.failed.connect(self.on_run_failed)
        self.runner.start()

    def on_run_finished(self):
        if os.path.exists("reconstructed.mat"):
            mat_data = loadmat("reconstructed.mat")
            recon = mat_data["recon_img"]
            if os.path.exists("metrics.json"):
                with open("metrics.json") as f:
                    metrics = json.load(f)
                psnr_val = metrics.get("psnr", 0)
                ssim_val = metrics.get("ssim", 0)
                self.status_label.setText(f"PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")
            else:
                self.status_label.setText("No metrics")
            plt.figure(figsize=(10, 4))
            plt.imshow(recon, cmap="gray")
            plt.title("Recon image")
            plt.axis("off")
            plt.show()
        else:
            self.status_label.setText("No recon image")

    def on_run_failed(self, error_msg):
        self.status_label.setText(f"Fail to run: {error_msg}")

    def closeEvent(self, event):
        temp_files = ["temp_input.mat", "sampling_mask.mat"]
        for file_path in temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CS_GUI()
    gui.show()
    sys.exit(app.exec_())
