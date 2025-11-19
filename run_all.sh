#!/bin/bash
#
# Master execution script for all ML models
# Run this from the ml_models directory
#

set -e  # Exit on error

echo "=========================================="
echo "ML Models Pipeline - Master Execution"
echo "=========================================="
echo ""

# Function to run and time a script
run_script() {
    script=$1
    echo ">>> Running: $script"
    start_time=$(date +%s)
    python "$script"
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "<<< Completed in ${elapsed}s"
    echo ""
}

# Check if we're in the right directory
if [ ! -f "utils.py" ]; then
    echo "ERROR: utils.py not found. Please run this script from the ml_models directory."
    exit 1
fi

# Group 1: LOOCV on Training Data
echo "=========================================="
echo "GROUP 1: LOOCV on Training Data"
echo "=========================================="
run_script "01_loocv_krr.py"
run_script "02_loocv_rf.py"
run_script "03_loocv_xgb.py"
run_script "04_oof_loocv_ensemble.py"
run_script "05_plot_loocv.py"

# Group 2: Repeated Train-Test Splits
echo "=========================================="
echo "GROUP 2: Repeated Train-Test Splits"
echo "=========================================="
run_script "06_repeated_krr.py"
run_script "07_repeated_rf.py"
run_script "08_repeated_xgb.py"
run_script "09_oof_repeated_ensemble.py"

# Group 3: Transfer Learning
echo "=========================================="
echo "GROUP 3: Transfer Learning"
echo "=========================================="
run_script "10_transfer_krr.py"
run_script "11_transfer_rf.py"
run_script "12_transfer_xgb.py"
run_script "13_oof_transfer_ensemble.py"
run_script "14_plot_transfer.py"

# Group 4: LOOCV on All Data
echo "=========================================="
echo "GROUP 4: LOOCV on All Data"
echo "=========================================="
run_script "15_alldata_loocv_krr.py"
run_script "16_alldata_loocv_rf.py"
run_script "17_alldata_loocv_xgb.py"
run_script "18_alldata_loocv_ensemble.py"
run_script "19_plot_alldata_loocv.py"

# Group 5: Repeated Train-Test on All Data
echo "=========================================="
echo "GROUP 5: Repeated Train-Test on All Data"
echo "=========================================="
run_script "20_alldata_repeated_krr.py"
run_script "21_alldata_repeated_rf.py"
run_script "22_alldata_repeated_xgb.py"
run_script "23_oof_alldata_repeated_ensemble.py"

echo "=========================================="
echo "ALL SCRIPTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Generated outputs:"
echo "  - 12 NPZ result files"
echo "  - 20 CSV hyperparameter files"
echo "  - 3 PNG comparison plots"
echo ""
echo "Check README.md for details on each output file."
