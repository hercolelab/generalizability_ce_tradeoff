import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
import numpy as np
from typing import Tuple
import os

def features_transformation(X_train, X_test, degree):

    from sklearn.preprocessing import PolynomialFeatures
    old_shape = X_train.shape
    poly = PolynomialFeatures(degree)
    poly.fit(X_train)
    X_train = poly.transform(X_train)
    X_test = poly.transform(X_test)
    print(f"Polynomial Features of degree {degree}. \nData from shape {old_shape} to shape {X_train.shape}.")
         
    return X_train, X_test

def preprocess(df, seed_split, degree, target_name):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.utils import resample, shuffle
    
    print("seed_split: ", seed_split)
    seed_resample = 42

    df = shuffle(df, random_state=seed_resample)

    # Define features and target
    X = df.drop(target_name, axis=1).values
    y = df[target_name].values

    test_size = 0.2
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed_split, stratify=df[target_name])


    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
 
    X_train, X_test = features_transformation(X_train, X_test, degree)
    
    #print(list(np.mean(X_train, axis = 0)))
    #print("\n","\n",list(np.std(X_train, axis = 0)))
    return X_train, X_test, y_train, y_test

def get_dataset(**kwargs) -> Tuple[TensorDataset, TensorDataset]:
    name: str = kwargs['name']
    binary_loss: bool = True
    seed_split: int = kwargs['seed_split']
    degree: int = kwargs['degree']
   
    dtype_in = torch.float32
    dtype_out = torch.float32 if binary_loss else torch.long
  

    if name == "water":
        
        try:
            df=pd.read_csv('data/water_potability.csv')
            
        except Exception:
            
            raise ValueError(f"Water dataset is not inside the data folder! cwd {os.getcwd()}")
        
        df['ph'] = df['ph'].fillna(value=df['ph'].median())
        df['Sulfate'] = df['Sulfate'].fillna(value=df['Sulfate'].median())
        df['Trihalomethanes'] = df['Trihalomethanes'].fillna(value=df['Trihalomethanes'].median())

        X_train, X_test, y_train, y_test = preprocess(df, seed_split, degree, 'Potability')
        
        train_set = TensorDataset(torch.tensor(X_train, dtype=dtype_in), torch.tensor(y_train,dtype=dtype_out))
        test_set = TensorDataset(torch.tensor(X_test, dtype=dtype_in), torch.tensor(y_test, dtype=dtype_out))
        
        return train_set, test_set
    
    else:
        
        raise ValueError(f"Dataset {name} is not available!")
            
    
    
if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo 
    
    # fetch dataset 
    ionosphere = fetch_ucirepo(id=52) 
    
    # data (as pandas dataframes) 
    X = ionosphere.data.features 
    y = ionosphere.data.targets 
    # Convert 'g' to 1 and 'b' to 0
    y = y.replace({'g': 1, 'b': 0})
    df = pd.concat([X, y], axis=1)

    dummy_param_bin = True
    dummy_param_param = {'resample': 1, 'poly_features_degree': 2, 'scaler': 'Standard', "seed_split":42}
    dtype_in = torch.float32
    dtype_out = torch.float32 if dummy_param_bin else torch.long


    X_train, X_test, y_train, y_test = preprocess(df, dummy_param_param, 'Class')
    print(y_test[y_test == 1].sum() / len(y_test))
    print(type(y_train), y_train.shape, y_train[0:5])
    print(dtype_out)

    train_set = TensorDataset(torch.tensor(X_train, dtype=dtype_in), torch.tensor(y_train,dtype=dtype_out))
    test_set = TensorDataset(torch.tensor(X_test, dtype=dtype_in), torch.tensor(y_test, dtype=dtype_out))

