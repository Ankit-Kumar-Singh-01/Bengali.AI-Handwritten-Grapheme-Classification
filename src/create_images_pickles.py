import pandas as pd
import joblib
import glob
from tqdm import tqdm

if __name__ == "__main__":
    files = glob.glob("../input/train_*.parquet")
    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        df = df.drop("image_id", axis=1) #dataframee will contain only pixel values
        image_array = df.values #put pixel values in numpy array of arrays, because it's v.slow to fetch from dataframe
        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_array[j, :], f"../input/image_pickles/{img_id}.pkl")