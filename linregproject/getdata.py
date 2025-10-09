import kagglehub
import os

download_path = os.path.join(os.getcwd(), "housedata") #download to housedata dir in current folder
os.makedirs(download_path, exist_ok=True)

# Download latest version to the local folder
path = kagglehub.dataset_download("shree1992/housedata", path=download_path)

print("Dataset downloaded to:", path)