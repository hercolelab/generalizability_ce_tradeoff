import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple
from src.dataset import get_dataset
from src.models import get_model
from src.trainer import LightningClassifier
from src.estimator import get_estimator
from src.evaluation import ClassifierEvaluator
from src.loss import get_loss
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
import argparse

def set_reproducibility(seed: int = 42) -> None:
    
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main() -> None:
        
    set_reproducibility()

    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--dataset_name', type=str, default='water', help='Dataset name')
    parser.add_argument('--seed_split', type=int, default=42, help='Seed for dataset split')
    parser.add_argument('--degree', type=int, default=1, help='Polynomial degree')
    parser.add_argument('--model_type', type=str, default='BMLP', help='Model type')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--radius', type=int, default=100, help='Radius parameter')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--distribution', type=str, default='uniform', help='Distribution type')
    parser.add_argument('--epochs', type=int, default=500, help='Max epochs for training')
    

    args = parser.parse_args()

    dataset_name = args.dataset_name
    seed_split = args.seed_split
    degree = args.degree
    model_type = args.model_type
    dropout = args.dropout
    n_samples = args.n_samples
    radius = args.radius
    distribution = args.distribution
    batch_size = args.batch_size
    epochs = args.epochs
    
    trainset, testset = get_dataset(name = dataset_name,seed_split = seed_split,degree = degree) 

    # TODO These preprocessing steps should ideally be refactored into get_dataset().
    # Extract the tensors from the trainset and testset
    train_data, train_targets = trainset[:][0], trainset[:][1]
    test_data, test_targets = testset[:][0], testset[:][1]
    print("train_data.shape: ", train_data.shape)
    # Apply PolynomialFeatures to the dataset
    #poly = PolynomialFeatures(degree=cfg.data.poly_degree)

    # Transform the data using PolynomialFeatures
    #train_data_poly = torch.tensor(poly.fit_transform(train_data.numpy()), dtype=torch.float32)
    #test_data_poly = torch.tensor(poly.transform(test_data.numpy()), dtype=torch.float32)

    # Create new TensorDatasets with the transformed data
    #trainset = TensorDataset(train_data_poly, train_targets)
    #testset = TensorDataset(test_data_poly, test_targets)

    # Update the input dimension in the model
    # cfg.data.input_dim = train_data_poly.size(1)
    
    input_dim = train_data.shape[1]
    
    if model_type == "BMLP":
        
        hidden_layers = [100, 30]
        
        model = get_model(model_type=model_type,
                          input_dim = input_dim, 
                          dropout=dropout, 
                          hidden_layers=hidden_layers)
    else:
        model = get_model(model_type="LogisticRegression", 
                          input_dim=input_dim)

    estimator = get_estimator(n_samples=n_samples, 
                              radius=radius, 
                              distribution=distribution, 
                              function=model, 
                              train_set=trainset)

    criterion = get_loss()
    evaluator = ClassifierEvaluator(classes=2)
    
    optimizer_configuration: dict[str, str|int|float] = {"name": "sgd", 
                                                         "lr": 0.001,
                                                         "weight_decay": 0.0}
    
    # set margin to False for BMLP
    clf =  LightningClassifier(model=model, 
                                criterion=criterion, 
                                optim_config=optimizer_configuration, 
                                evaluator=evaluator, 
                                estimator=estimator, 
                                margin = False)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    trainer_params: dict[str, bool|str|int|float] = {"enable_progress_bar": False, 
                                                     "accelerator": "gpu", 
                                                     "num_sanity_val_steps": 0, 
                                                     "max_epochs": epochs}
    csv_logger = CSVLogger(
        save_dir="log/",        # e.g. "csv_logs/"
        name=f"{dataset_name}_{model_type}"
    )
    pl.Trainer()
    trainer = pl.Trainer(**trainer_params, logger=csv_logger)
    trainer.fit(clf, train_loader, test_loader)

if __name__ == "__main__":
    print("CUDA version PyTorch was built with:", torch.version.cuda)
    print("Is CUDA available?:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA runtime version:", torch.version.cuda)
        print("Torch is using device:", torch.cuda.current_device())
    main()