# Makefile for setting up the Conda environment for the KRR LOOCV project.
#
# This version uses a single, robust conda command to install all dependencies,
# including the CUDA toolkit and related libraries, as this is the most
# reliable method.

# --- Configuration ---
# Set the name of your conda environment
ENV_NAME := m_leeching
# Set the Python version
PYTHON_VERSION := 3.10

# --- Phony targets prevent conflicts with files of the same name ---
.PHONY: all install create_env clean help

# --- Default target ---
all: install

# --- Help target to display available commands ---
help:
	@echo "Available commands:"
	@echo "  make install    - Creates the Conda environment and installs all dependencies."
	@echo "  make clean      - Removes the Conda environment."
	@echo "  make help       - Shows this help message."

# --- Environment Creation ---
# Creates the conda environment if it doesn't exist.
create_env:
	@echo ">>> Creating Conda environment '$(ENV_NAME)' with Python $(PYTHON_VERSION)..."
	@conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y --quiet
	@echo ">>> Environment created successfully."

# --- Installation ---
# Installs all required packages in a single, robust command.
install: create_env
	@echo ">>> Installing all packages from conda-forge (this can take several minutes)..."
	@conda install -n $(ENV_NAME) -c conda-forge --yes \
		cupy \
		cudnn \
		cutensor \
		nccl \
		pandas \
		numpy \
		matplotlib \
		tqdm \
		"scikit-learn>=1.2" \
		adjusttext \
		openpyxl
	@echo "✅ Installation complete."
	@echo "To run your script, use:"
	@echo "   conda activate $(ENV_NAME)"
	@echo "   python KRR_all_qml2.py MS_samples.xlsx"

# --- Cleanup ---
# Removes the conda environment entirely.
clean:
	@echo ">>> Removing Conda environment '$(ENV_NAME)'..."
	@conda env remove --name $(ENV_NAME)
	@echo ">>> Environment removed successfully."


