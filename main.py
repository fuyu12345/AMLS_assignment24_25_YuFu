import subprocess
import os
# 12
if __name__ == "__main__":
    # Get the path to A.py
    # A_logistic_regression_combined_dataset
    a_script_path = os.path.join(os.path.dirname(__file__), "B", "CNN.py")
    
    # Run A.py using subprocess
    print(f"Running {a_script_path}...")
    subprocess.run(["python", a_script_path])
