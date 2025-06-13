import torch
from src.estimator import MontecarloEstimator
import pytorch_lightning as L
from src.evaluation import ClassifierEvaluator
from src.optimizer import get_optimizer
import numpy as np
import inspect
import time

def get_trainer(type: str, model, criterion, evaluator, config,  estimator = None):
    
    # if "regularized" in type:
    #     from src.trainer.trainer import CounterfactualLightningClassifier

    #     return CounterfactualLightningClassifier(model=model,
    #                                              criterion=criterion,
    #                                              config=config,
    #                                              evaluator=evaluator,
    #                                              estimator=estimator)
    
    # elif type == "normal":
    from src.trainer import LightningClassifier

    return LightningClassifier(model=model,
                                   criterion=criterion,
                                   config=config,
                                   evaluator=evaluator)
    
    

class LightningClassifier(L.LightningModule):
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 criterion: torch.nn.Module,
                 optim_config: dict,
                 evaluator: ClassifierEvaluator,
                 estimator: MontecarloEstimator,
                 margin: bool) -> None:
        
        super().__init__()

        self.model = model
        self.optim_config = optim_config
        self.criterion = criterion
        self.train_output = []
        self.train_target = []
        self.train_loss = []
        self.train_p_x = []
        self.val_output = []
        self.val_target = []
        self.val_loss = []
        self.val_p_x = []
        self.train_embeddings = []
        self.test_embeddings = []
        self.evaluator = evaluator
        self.estimator = estimator
        self.show_embedding = False
        self.margin = margin
        


#    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, optimizer_idx: int, *args, **kwargs):
#        super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)
#        for name, param in self.model.named_parameters():
#            if param.grad is not None:
#                print(f"{name}: {param.grad.norm()}")  # Print gradient norm
#            else:
#                print(f"{name}: No gradient found")    
    
    def configure_optimizers(self):
        
        return get_optimizer(params=self.model.parameters(), config=self.optim_config)
        
        
    def on_train_epoch_start(self) -> None:
        self.train_t_start = time.time_ns()
        self.train_output = []
        self.train_target = []
        self.train_loss = []
        self.train_estimate = []
        self.train_margin = []
    
    def on_train_epoch_end(self) -> None:
        self.train_t_end = time.time_ns()
        stage: str = "train"
        with torch.no_grad():
            accuracy, f1, precision, recall, crossentropy = self.evaluator.get_complete_evaluation(self.train_output, self.train_target)
        if self.margin:
            evcp_bound = self.evaluator.get_avg_evcp_bound(np.mean(self.train_margin), self.estimator.radius, 5005) if np.mean(self.train_margin) <= self.estimator.radius else 0
   
    
        log_data = {
            f"{stage}/loss": sum(self.train_loss) / len(self.train_loss),
            f"{stage}/epoch": self.current_epoch,
            f"{stage}/accuracy": accuracy,
            f"{stage}/f1-score": f1,
            f"{stage}/precision": precision,
            f"{stage}/recall": recall,
            f"{stage}/crossentropy": crossentropy,
            f"{stage}/time_elapsed" : self.train_t_end - self.train_t_start
        }

        if self.margin:
            log_data.update(
                {f"{stage}/avgmargin" : np.mean(self.train_margin),
                f"{stage}/avgevcpbound": evcp_bound}
            )

        estimator_log_data = self.estimator.build_log(self.train_estimate, stage)

        log_data.update(estimator_log_data)

        self.log_dict(log_data, on_epoch=True, on_step=False) 
        
        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:

        data, target = batch
        output = self.model(data)
        
        #out, target_cf = self.estimator.get_counterfactual(data, output, grad=self.counterfactual)
        #p_x = self.estimator.counterfactual_probability(out=out, target=target_cf)
        #if self.counterfactual:
        #    values = values | { "out_cf": out, "target_cf": target_cf}
        if (self.current_epoch-1) % 60 == 0:
            estimate = self.estimator.get_estimate(data = data, output = output)
        else: 
            estimate = torch.zeros(data.shape[0], dtype=torch.float32)
        values: dict = {"input": output, "target": target, "estimate": estimate, "weights": self.model.parameters(), "data": data}

        #forward_signature = list(inspect.signature(self.criterion.__class__.forward).parameters.keys())[1:] # the first parameter is self, so it can be dropped
        #values = {key: value for key,value in values.items() if key in forward_signature}


        torch.set_grad_enabled(mode=True)
        loss = self.criterion(**values)       
        #print("batch_idx:", batch_idx)
        #for name, param in self.model.named_parameters():
        #    if param.grad is not None:
        #        print(f"{name}: {param.grad.norm()}")
        #    else:
        #        print(f"{name}: No gradient found")
      
        self.train_target += target.tolist()
        self.train_output += output.tolist()
        self.train_loss += [loss.item()]
        self.train_estimate += estimate.tolist()

        if self.margin:
            torch.set_grad_enabled(mode=False)
            w = self.model.linear.weight.cpu()
            f_x = self.model.forward(data).cpu()
            #print("f_x.shape:", f_x.shape)
            #print("w.shape:", w.shape)
            margin = np.abs(f_x/np.linalg.norm(w))
            self.train_margin += margin.tolist()
            #print("margin:", margin.shape)
            torch.set_grad_enabled(mode=True)

        return loss
    
    def on_validation_epoch_start(self) -> None:
        self.val_t_start = time.time_ns()
        self.val_output = []
        self.val_target = []
        self.val_loss = []
    
    
    def on_validation_epoch_end(self) -> None:
        self.val_t_end = time.time_ns()
        if self.trainer.state.stage != "sanity_check":
            
            stage: str = "validation"
            accuracy, f1, precision, recall, crossentropy = self.evaluator.get_complete_evaluation(self.val_output, self.val_target)
            
            log_data = {
                f"{stage}/loss": sum(self.val_loss) / len(self.val_loss),
                f"{stage}/epoch": self.current_epoch,
                f"{stage}/accuracy": accuracy,
                f"{stage}/f1-score": f1,
                f"{stage}/precision": precision,
                f"{stage}/recall": recall,
                f"{stage}/crossentropy": crossentropy,
                f"{stage}/time_elapsed" : self.val_t_end - self.val_t_start
            }

            #estimator_log_data = self.estimator.build_log(self.val_estimate, stage)

            #log_data.update(estimator_log_data)

            self.log_dict(log_data, on_epoch=True, on_step=False) 

    def validation_step(self, batch, batch_idx):
        
  
        data, target = batch
        output = self.model(data)
        #values: dict = {"input": output, "target": target}
        #out, target_cf = self.estimator.get_counterfactual(data, output, grad=False)
        #p_x = self.estimator.get_estimate(out=out, target=target_cf)
   
#        old_params = {name: param.clone() for name, param in self.model.named_parameters()}
        #torch.set_grad_enabled(mode=True)
        #estimate = self.estimator.get_estimate(data = data, output = output)
   
        #torch.set_grad_enabled(mode=False)
        #  new_params = {name: param for name, param in self.model.named_parameters()}

        values: dict = {"input": output, "target": target, "estimate": "None", "weights": self.model.parameters(), "data": data}
#        forward_signature = list(inspect.signature(self.criterion.__class__.forward).parameters.keys())[1:] # the first parameter is self, so it can be dropped
#        values = {key: value for key,value in values.items() if key in forward_signature}
        #if self.counterfactual:
        #    values = values | { "out_cf": out, "target_cf": target_cf}        
     
        val_loss = self.criterion(**values)   
        self.val_target += target.tolist()
        self.val_output += output.tolist()
        self.val_loss += [val_loss.item()]   
    

#        for name in old_params:
#            if not torch.equal(old_params[name], new_params[name].data):
#                print(f"Parameter '{name}' has changed.")
#            else:
#                print(f"Parameter '{name}' is unchanged.")
        return val_loss
