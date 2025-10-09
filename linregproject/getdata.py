from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi(); api.authenticate()

#new folder to save data too in dir
download_path = os.path.join(os.getcwd(), "housedata")
os.makedirs(download_path, exist_ok=True)
api.dataset_download_files("shree1992/housedata", path=download_path, unzip=True)

print("Path to dataset files:", download_path)