import sys
import json
import os
import subprocess
import matplotlib.pyplot as plt
from scipy.io import loadmat

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QHBoxLayout
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
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        # 算法选择
        self.alg_selector = QComboBox()
        self.alg_selector.addItems(["cvx", "BSBL_FM"])
        self.alg_selector.currentTextChanged.connect(self.update_param_visibility)

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

        # Patch size
        self.patch_label = QLabel("Patch size:")
        self.patch_size_input = QSpinBox()
        self.patch_size_input.setRange(8, 512)
        self.patch_size_input.setValue(64)

        # Stride
        self.stride_label = QLabel("Stride:")
        self.stride_input = QSpinBox()
        self.stride_input.setRange(1, 512)
        self.stride_input.setValue(8)

        # Slice index（初始隐藏）
        self.slice_label = QLabel("Slice index:")
        self.slice_index_input = QSpinBox()
        self.slice_index_input.setRange(0, 999)
        self.slice_index_input.setValue(0)
        self.slice_label.setVisible(False)
        self.slice_index_input.setVisible(False)

        # ROI 选择
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

        # BSBL 专用参数
        self.snr_label = QLabel("SNR:")
        self.snr_input = QSpinBox()
        self.snr_input.setRange(0, 100)
        self.snr_input.setValue(30)

        self.blklen_label = QLabel("Blk_len:")
        self.blklen_input = QSpinBox()
        self.blklen_input.setRange(1, 64)
        self.blklen_input.setValue(8)

        self.lambda_label = QLabel("Learn Lambda:")
        self.lambda_selector = QComboBox()
        self.lambda_selector.addItems(["No learning rule", "Medium SNR", "High SNR"])

        # 按钮
        self.load_button = QPushButton("Choose your HDF5 file")
        self.load_button.clicked.connect(self.load_file)

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

        self.layout.addWidget(self.patch_label)
        self.layout.addWidget(self.patch_size_input)
        self.layout.addWidget(self.stride_label)
        self.layout.addWidget(self.stride_input)

        self.layout.addWidget(self.slice_label)
        self.layout.addWidget(self.slice_index_input)

        self.layout.addWidget(self.use_full_image_checkbox)
        self.layout.addWidget(self.roi_label)
        self.layout.addLayout(roi_layout)

        self.layout.addWidget(self.snr_label)
        self.layout.addWidget(self.snr_input)
        self.layout.addWidget(self.blklen_label)
        self.layout.addWidget(self.blklen_input)
        self.layout.addWidget(self.lambda_label)
        self.layout.addWidget(self.lambda_selector)

        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.status_label)

        self.setLayout(self.layout)
        self.update_param_visibility("cvx")

    def toggle_roi_inputs(self, state):
        visible = not self.use_full_image_checkbox.isChecked()
        self.roi_label.setVisible(visible)
        self.x_start_input.setVisible(visible)
        self.x_end_input.setVisible(visible)
        self.y_start_input.setVisible(visible)
        self.y_end_input.setVisible(visible)

    def update_param_visibility(self, algo):
        is_bsbl = (algo == "BSBL_FM")
        self.snr_label.setVisible(is_bsbl)
        self.snr_input.setVisible(is_bsbl)
        self.lambda_label.setVisible(is_bsbl)
        self.lambda_selector.setVisible(is_bsbl)
        self.blklen_label.setVisible(is_bsbl)
        self.blklen_input.setVisible(is_bsbl)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose your HDF5 file", "", "HDF5 Files (*.hdf5)")
        if file_path:
            self.image_path = file_path
            try:
                shape = get_hdf5_image_dims(file_path)
                ndim = len(shape)
                self.image_shape = shape

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

                # ROI 范围设置
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

        # ROI
        if not self.use_full_image_checkbox.isChecked():
            cfg["roi"] = {
                "x_start": self.x_start_input.value(),
                "x_end": self.x_end_input.value(),
                "y_start": self.y_start_input.value(),
                "y_end": self.y_end_input.value()
            }

        if algorithm == "BSBL_FM":
            cfg["snr"] = self.snr_input.value()
            cfg["learn_lambda"] = self.lambda_selector.currentIndex()
            cfg["blk_len"] = self.blklen_input.value()

        with open("config.json", "w") as f:
            json.dump(cfg, f)

        # 生成 mask
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
                    print(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CS_GUI()
    gui.show()
    sys.exit(app.exec_())
