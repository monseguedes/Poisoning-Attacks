"""
This file is to select a subset of features to then give to the
polynomial optimization problem. This means that categorical features
have at most 3 categories.
"""

import pandas as pd
import os

def choose_data(max_num_categories : int, dataset_name: str):

    df = pd.read_csv("data/" + dataset_name + "/data-binary.csv")

    # Choose only categorical features with less than max_num_categories categories
    categorical_columns = [column for column in df.columns if ":" in column]
    categorical_names = list(dict.fromkeys([name.split(":")[0] for name in categorical_columns]))
    categories_dict = {
        cat_name: len([
            category.split(":")[1]
            for category in categorical_columns
            if category.startswith(cat_name + ":")
        ])
        for cat_name in categorical_names
    }
  
    # Filter dictionary to those values less than max_num_categories
    categories_dict = {k: v for k, v in categories_dict.items() if v <= max_num_categories}

    # Pick only columns with these categorical features (keys) form dataframe
    df = df[[column for column in df.columns if column.split(":")[0] in categories_dict.keys() or ":" not in column]]

    # Create a dictionary that maps keys of categories_dict to 0, 1, 2, ...
    categrical_features = categories_dict.keys()
    map = {feature + ":" + str(category) : str(i) + ":" + str(category) for i, feature in enumerate(categrical_features) for category in range(categories_dict[feature])}

    # Rename first part of column names using map
    df = df.rename(columns=map)

    # Save dataframe to csv
    # Create folder if not there
    try:
        os.mkdir("data/" + dataset_name + str(max_num_categories))
    except:
        pass
    
    df.to_csv("data/" + dataset_name + str(max_num_categories) + "/data-binary.csv", index=False)
        

print(choose_data(4, "10num10cat"))