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



def main() -> None:
            np.random.seed(42) 
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            

            dataset_name = "water"
            seed_split = 42
            degree = 1
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
          
            model_type = "BMLP"
            if model_type == "BMLP":
                input_dim = train_data.shape[1]
                dropout = 0.0
                hidden_layers = [100, 30]
                model = get_model(model_type = model_type, input_dim = input_dim, dropout = dropout, hidden_layers = hidden_layers)
            else:
                input_dim = train_data.shape[1]
                model = get_model(model_type = "LogisticRegression", input_dim = input_dim)

            estimator = get_estimator(n_samples = 1000, radius = 100, distribution = "uniform", function = model, train_set = trainset)

            criterion = get_loss()
            evaluator = ClassifierEvaluator(classes=2)
            
            # set margin to False for BMLP
            clf =  LightningClassifier(model=model, 
                                       criterion=criterion, 
                                       optim_config={"name": "sgd", "lr": 0.001,"weight_decay": 0.0}, 
                                       evaluator=evaluator, 
                                       estimator=estimator, 
                                       margin = False)
                
            train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
            test_loader = DataLoader(testset, batch_size=128, shuffle=True)
            
            trainer_params = {"enable_progress_bar": False, "accelerator": "gpu", "num_sanity_val_steps": 0, "max_epochs": 3}
            csv_logger = CSVLogger(
                save_dir="log/",        # e.g. "csv_logs/"
                name=f"{dataset_name}_{model_type}"
            )
            pl.Trainer()
            trainer = pl.Trainer(**trainer_params, logger=csv_logger)
            trainer.fit(clf, train_loader, test_loader)

if __name__ == "__main__":
        print("CUDA version PyTorch was built with:", torch.version.cuda)

# Shows whether CUDA is available and what device is being used
print("Is CUDA available?:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA runtime version:", torch.version.cuda)
    print("Torch is using device:", torch.cuda.current_device())
main()