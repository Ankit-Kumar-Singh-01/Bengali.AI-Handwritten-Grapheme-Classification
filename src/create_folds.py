import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    print(df.head())

    #We create a column kfold and fill it with -1
    df.loc[:, 'kfold'] = -1
    print(df.head())
    #Shuffling of dataset(just a good idea)
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())

    X = df.image_id.values
    y = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    mskf = MultilabelStratifiedKFold(n_splits=5) #5-folds

    #get the training and validation indices from mskf
    for fold, (trn_, val_) in enumerate(mskf.split(X,y)):
        print("TRAIN: ", trn_, "VAL: ", val_) #Print training & validation indices for every fold
        df.loc[val_, 'kfold'] = fold
    
    print(df.kfold.value_counts())
    df.to_csv("../input/train_folds.csv", index=False)