import kagglehub

# Download latest version
path = kagglehub.dataset_download("adhoppin/blood-cell-detection-datatset")

print("Path to dataset files:", path)