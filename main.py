import subprocess
import os

if __name__ == "__main__":
    # Get the path to A.py
    a_script_path = os.path.join(os.path.dirname(__file__), "A", "A_logistic_regression_combined_dataset.py")
    
    # Run A.py using subprocess
    print(f"Running {a_script_path}...")
    subprocess.run(["python", a_script_path])