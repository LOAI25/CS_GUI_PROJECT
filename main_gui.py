import sys
import json
import os
import subprocess
import matplotlib.pyplot as plt
from scipy.io import loadmat

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QComboBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import QThread, pyqtSignal

from algorithms.common import get_hdf5_image_dims

# Prevent GUI from freezing or hanging on close while the algorithm is running
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
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.alg_selector = QComboBox()
        self.alg_selector.addItems(["cvx", "BSBL_FM"])
        self.alg_selector.currentTextChanged.connect(self.update_param_visibility)

        self.sampling_label = QLabel("sampling rate:")
        self.sampling_input = QDoubleSpinBox()
        self.sampling_input.setRange(0.01, 1.0)
        self.sampling_input.setSingleStep(0.05)
        self.sampling_input.setValue(0.3)

        self.block_label = QLabel("block size:")
        self.block_size_input = QSpinBox()
        self.block_size_input.setRange(8, 512)
        self.block_size_input.setValue(64)

        self.slice_label = QLabel("slice index:")
        self.slice_index_input = QSpinBox()
        self.slice_index_input.setRange(0, 999)
        self.slice_index_input.setValue(0)

        self.snr_label = QLabel("SNR:")
        self.snr_input = QSpinBox()
        self.snr_input.setRange(0, 100)
        self.snr_input.setValue(30)

        self.blklen_label = QLabel("blk_len:")
        self.blklen_input = QSpinBox()
        self.blklen_input.setRange(1, 64)
        self.blklen_input.setValue(8)

        self.lambda_label = QLabel("Learn Lambda:")
        self.lambda_selector = QComboBox()
        self.lambda_selector.addItems(["No learning rule", "Medium SNR", "High SNR"])

        self.load_button = QPushButton("Choose your HDF5 file")
        self.load_button.clicked.connect(self.load_file)

        self.run_button = QPushButton("Do compressed sensing")
        self.run_button.clicked.connect(self.run_cs)

        self.status_label = QLabel("The file hasn't been loaded")

        self.layout.addWidget(QLabel("Choose your algorithm:"))
        self.layout.addWidget(self.alg_selector)

        self.layout.addWidget(self.sampling_label)
        self.layout.addWidget(self.sampling_input)
        self.layout.addWidget(self.block_label)
        self.layout.addWidget(self.block_size_input)

        self.layout.addWidget(self.slice_label)
        self.layout.addWidget(self.slice_index_input)

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

        self.slice_label.setVisible(False)
        self.slice_index_input.setVisible(False)

    # Show/hide BSBL_FM-specific parameters depending on selected algorithm
    def update_param_visibility(self, algo):
        is_bsbl = (algo == "BSBL_FM")
        self.snr_label.setVisible(is_bsbl)
        self.snr_input.setVisible(is_bsbl)
        self.blklen_label.setVisible(is_bsbl)
        self.blklen_input.setVisible(is_bsbl)
        self.lambda_label.setVisible(is_bsbl)
        self.lambda_selector.setVisible(is_bsbl)

    # Automatically check image dimensionality and update slice index input
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose your HDF5 file", "", "HDF5 Files (*.hdf5)")
        if file_path:
            self.image_path = file_path
            try:
                shape = get_hdf5_image_dims(file_path)
                ndim = len(shape)

                if ndim == 2:
                    self.slice_label.setVisible(False)
                    self.slice_index_input.setVisible(False)
                    self.slice_index_input.setMaximum(0)
                elif ndim == 3:
                    self.slice_label.setVisible(True)
                    self.slice_index_input.setVisible(True)
                    self.slice_index_input.setMaximum(shape[2] - 1)
                else:
                    self.status_label.setText(f"Wrong image dim: {shape}")
                    return

                self.status_label.setText(f"File is loaded with dim: {shape}")

            except Exception as e:
                self.status_label.setText(f"The file is fail to load: {e}")

    def run_cs(self):
        if not self.image_path:
            self.status_label.setText("The file hasn't been selected")
            return

        algorithm = self.alg_selector.currentText()
        cfg = {
            "image_path": self.image_path,
            "slice_index": self.slice_index_input.value(),
            "sampling_rate": self.sampling_input.value(),
            "block_size": self.block_size_input.value(),
            "algorithm": algorithm,
            "output_path": "reconstructed.mat",
            "metrics_path": "metrics.json"
        }

        if algorithm == "BSBL_FM":
            cfg["snr"] = self.snr_input.value()
            cfg["blk_len"] = self.blklen_input.value()
            cfg["learn_lambda"] = self.lambda_selector.currentIndex()

        with open("config.json", "w") as f:
            json.dump(cfg, f)

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

    # Ensure temp_input.mat is deleted when the GUI window is closed, even if the algorithm was still running
    def closeEvent(self, event):
        temp_mat_path = "temp_input.mat"
        if os.path.exists(temp_mat_path):
            try:
                os.remove(temp_mat_path)
                print(f"Delete temporary file: {temp_mat_path}")
            except Exception as e:
                print(f"Fail to delete: {e}")
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CS_GUI()
    gui.show()
    sys.exit(app.exec_())
