import os
import subprocess

def main():
    subprocess.run(["python", "scripts/data_preparation.py"])
    subprocess.run(["python", "scripts/feature_engineering.py"])
    subprocess.run(["python", "scripts/model_training.py"])
    subprocess.run(["python", "scripts/evaluation.py"])

if __name__ == "__main__":
    main()
