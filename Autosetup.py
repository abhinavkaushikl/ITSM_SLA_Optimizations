import os
import shutil

# Define the folder structure
folders = [
    "mlops_project/configs",
    "mlops_project/data/raw",
    "mlops_project/data/interim",
    "mlops_project/data/processed",
    "mlops_project/data/external",
    "mlops_project/data_ingestion",
    "mlops_project/data_validation",
    "mlops_project/feature_engineering",
    "mlops_project/models/artifacts",
    "mlops_project/evaluation/metrics",
    "mlops_project/evaluation/reports",
    "mlops_project/inference",
    "mlops_project/deployment/k8s",
    "mlops_project/deployment/cicd",
    "mlops_project/monitoring",
    "mlops_project/notebooks",
    "mlops_project/utils",
    "mlops_project/tests"
]

# Create the folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create base files
base_files = [
    "mlops_project/requirements.txt",
    "mlops_project/.env",
    "mlops_project/.gitignore",
    "mlops_project/README.md"
]

for file in base_files:
    with open(file, 'w') as f:
        f.write("")

# Zip the folder structure
shutil.make_archive("mlops_project_template", "zip", "mlops_project")

print("✅ MLOps folder structure created and zipped as 'mlops_project_template.zip'")
