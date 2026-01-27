This project analyzes a large-scale Steam reviews dataset. Because the raw data is ~2GB, you will need to fetch and extract it locally using the steps below.

---

## Quick Start

### Fetching Data
Run this command to pull the data from Kaggle using curl:
```bash
curl -L -o steam-reviews.zip "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/steam-reviews"
```

### Extracting Data
Run this command to extract the data into the ./data folder in the /nlptest directory
Please note that this command only needs to be run once
```bash
python3 getdata.py
```