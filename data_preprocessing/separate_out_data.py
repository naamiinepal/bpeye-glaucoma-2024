import pandas as pd
import os 
import shutil 
from glob import glob 
import argparse
    

def preprocess_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder") #/mnt/Enterprise/data which contains train.csv file
    parser.add_argument("output_folder") # make output_data folder outside the data_folder
    args = parser.parse_args()

    df = pd.read_csv(f'{args.data_folder}/train_labels.csv')

    os.makedirs(os.path.join(args.output_folder, "preprocessed_overall_data", "RG"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "preprocessed_overall_data", "NRG"), exist_ok=True)
    
    for img_path in sorted(glob(f'{args.data_folder}/*/*/*.jpg')):    # output folder should be outside, becoz this takeds all elements of data folder
        # print(str(os.path.split(img_path)[1]).split('.jpg')[0])
        # print(df.loc[df['challenge_id'] == os.path.split(img_path)[1].split('.jpg')[0], 'class'].item())
        if df.loc[df['challenge_id'] == os.path.split(img_path)[1].split('.jpg')[0], 'class'].item() == 'NRG':
            print(os.path.join(f'{args.output_folder}/preprocessed_overall_data/NRG/', os.path.split(img_path)[1]))
            shutil.copy(img_path, os.path.join(f'{args.output_folder}/preprocessed_overall_data/NRG/', os.path.split(img_path)[1])) 

        elif df.loc[df['challenge_id'] == os.path.split(img_path)[1].split('.jpg')[0], 'class'].item() == 'RG':
            print(os.path.join(f'{args.output_folder}/preprocessed_overall_data/RG/', os.path.split(img_path)[1]))
            shutil.copy(img_path, os.path.join(f'{args.output_folder}/preprocessed_overall_data/RG/', os.path.split(img_path)[1])) 
        

if __name__ == "__main__":
    preprocess_data()