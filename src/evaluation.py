from torchmetrics import Accuracy, F1Score, Precision, Recall
import torch

class ClassifierEvaluator:
    
    def __init__(self, classes: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

        self.accuracy = Accuracy(task="multiclass", num_classes=classes).to(self.device)
        self.f1 = F1Score(task="multiclass", num_classes=classes).to(self.device)
        self.precision = Precision(task="multiclass", average='macro', num_classes=classes).to(self.device)
        self.recall = Recall(task="multiclass", average='macro', num_classes=classes).to(self.device)
        self.crossentropy = torch.nn.functional.binary_cross_entropy_with_logits
        
    
    def get_complete_evaluation(self, output, target):
        
        output = torch.tensor(output, device=self.device)
        target = torch.tensor(target, device=self.device)

        crossentropy = self.crossentropy(output, target)
        
        output = torch.argmax(output, dim=-1) if output.ndim > 1 else (output > 0.5).float()
        
        accuracy = self.accuracy(output, target)
        f1 = self.f1(output, target)
        precision = self.precision(output, target)
        recall = self.recall(output, target)
       
        
        return accuracy, f1, precision, recall, crossentropy
    

    def get_avg_evcp_bound(self, margin, epsilon, n):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.special import betainc

        def stable_prefactor(gamma, epsilon, n, thresh=-50.0):
            """
            Compute ε^n / (ε^n - γ^n) in a numerically stable way:
            = 1 / (1 - (γ/ε)^n)
            = 1 / (1 - exp(n * log(γ/ε)))
            For logt < thresh, exp(logt) is negligible and ratio ≈ 1.
            
            Parameters
            ----------
            gamma : array_like
                γ values, 0 < γ < ε.
            epsilon : float
                ε > 0.
            n : float
                Exponent n.
            thresh : float, optional
                Log‐threshold below which exp(logt) is treated as zero.
            
            Returns
            -------
            ratio : ndarray
                The stable prefactor, same shape as `gamma`.
            """
            gamma = np.asarray(gamma, dtype=float)
            logt = n * np.log(gamma / epsilon)                       # natural log, elementwise :contentReference[oaicite:0]{index=0}
            # Where logt is very small, set ratio = 1.0
            small_mask = logt < thresh                              # boolean mask, elementwise comparison :contentReference[oaicite:1]{index=1}
            ratio = np.empty_like(logt)
            # For tiny exp(logt), use 1.0 directly
            ratio[small_mask] = 1.0
            # Else compute 1 / (1 - exp(logt))
            ratio[~small_mask] = 1.0 / (1.0 - np.exp(logt[~small_mask]))  # safe exp for moderate arguments :contentReference[oaicite:2]{index=2}
            return ratio


        def p_i_epsilon(gamma_i, epsilon, n):
            """
            Compute p_i^ε = 1/2 * ε^n / (ε^n - γ_i^n) * I(1 - (γ_i/ε)^2; (n+1)/2, 1/2)
            
            Parameters
            ----------
            gamma_i : float or array_like
                Values of γ_i, must satisfy 0 < γ_i < ε.
            epsilon : float
                The ε parameter, must be > 0.
            n : float
                The exponent n.
            
            Returns
            -------
            p : float or ndarray
                The computed p_i^ε values, same shape as `gamma_i`.
            """
            gamma_i = np.asarray(gamma_i, dtype=float)
            
            # Check domain
            if np.any(gamma_i <= 0) or epsilon <= 0:
                raise ValueError("Require ε > 0 and γ_i > 0")
            if np.any(gamma_i >= epsilon):
                raise ValueError("Require γ_i < ε")
            
            # Compute the argument x of the regularized incomplete beta
            x = 1.0 - (gamma_i / epsilon)**2
            
            # Parameters a and b for I_x(a, b)
            a = (n + 1.0) / 2.0
            b = 0.5
            
            # Regularized incomplete beta I_x(a, b)
            I = betainc(a, b, x)
            
            # Prefactor
            #prefac = 0.5 * epsilon**n / (epsilon**n - gamma_i**n)
            prefac = 0.5 * stable_prefactor(gamma_i, epsilon, n)  # use stable prefactor :contentReference[oaicite:5]{index=5}

            return prefac * I
        
        return p_i_epsilon(margin, epsilon, n)