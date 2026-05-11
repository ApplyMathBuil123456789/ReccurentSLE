# -*- coding: utf-8 -*-
"""REcurent_NNN

"""

# -*- coding: utf-8 -*-

"""
Rekurrent Signal Denoising Network (SLE-RNN)
=============================================

Recurrent neural network for iterative signal reconstruction.
At each iteration, a clean signal is separated from a noisy signal and
only the reconstructed signal is passed to the next iteration.

Mathematical model:
Y^(0) = Y_noisy (input)
S^(t) = Filter(Y^(t-1); α, κ, r) (spectral reconstruction)
Y^(t) = S^(t) (noiseless, only reconstructed signal)
result: S^(T) (after T iterations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURATION AND HELPER CLASSES
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration"""
    M: int = 1000                     # Signal length
    T: int = 1                        # Time interval [0, T]
    epsilon: float = 0.01             # Noise level
    num_iterations: int = 3           # Number of recurrent iterations

    # Gating network architecture
    gating_hidden: int = 64

    # Training parameters
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-2


@dataclass
class ParameterGrid:
    """Save adaptive parameter grid"""
    alphas: List[float]
    kappas: List[float]
    rs: List[float]

    def __len__(self) -> int:
        return len(self.alphas) * len(self.kappas) * len(self.rs)

    def get_all_combinations(self) -> List[Tuple[float, float, float]]:
        """Returns all combinations of (alpha, kappa, r)"""
        import itertools
        return list(itertools.product(self.alphas, self.kappas, self.rs))


# =============================================================================
# 2. SIGNAL GENERATOR
# =============================================================================

class SignalGenerators:
    """Various alarm functions"""

    @staticmethod
    def sinusoidal(t: np.ndarray) -> np.ndarray:
        A = random.uniform(0.5, 1.0)
        f = random.uniform(0.1, 1.0)
        phi = random.uniform(0, 2 * np.pi)
        return A * np.sin(2 * np.pi * t)

    @staticmethod
    def square_wave(t: np.ndarray) -> np.ndarray:
        A = random.uniform(0.1, 2.0)
        phi = random.uniform(0, 2 * np.pi)
        s = A * np.sign(np.sin(2 * np.pi * t + phi))
        s[-1] = 0  # Oxirgi nuqtani 0 qilish
        return s

    @staticmethod
    def sawtooth(t: np.ndarray) -> np.ndarray:
        A = random.uniform(0.1, 2.0)
        return A * (t - np.floor(t + 0.5))

    @staticmethod
    def triangular(t: np.ndarray) -> np.ndarray:
        A = random.uniform(0.3, 1.0)
        return A * np.abs(2 * (t - np.floor(t + 0.5)))

    @staticmethod
    def damped_oscillation(t: np.ndarray) -> np.ndarray:
        A = random.uniform(0.5, 2.0)
        alpha = random.uniform(1.0, 5.0)
        f = random.uniform(1.0, 5.0)
        return A * np.exp(-alpha * t) * np.sin(2 * np.pi * f * t)

    @staticmethod
    def gaussian_pulse(t: np.ndarray) -> np.ndarray:
        A = random.uniform(0.5, 2.0)
        mu = random.uniform(0.3, 0.7)
        sigma = random.uniform(0.05, 0.2)
        return A * np.exp(-(t - mu)**2 / (2 * sigma**2))

    @staticmethod
    def chirp(t: np.ndarray) -> np.ndarray:
        A = random.uniform(0.5, 2.0)
        f0 = random.uniform(1.0, 5.0)
        k = random.uniform(1.0, 4.0)
        return A * np.sin(2 * np.pi * (f0 * t + (k / 2) * t**2))

    @staticmethod
    def single_pulse(t: np.ndarray) -> np.ndarray:
        pulse_width = random.uniform(0.05, 0.2)
        t_start = random.uniform(0, 1 - pulse_width)
        signal = np.zeros_like(t)
        signal[(t >= t_start) & (t < t_start + pulse_width)] = 1.0
        return signal

    @classmethod
    def get_all_generators(cls) -> List[Callable]:
        return [
            cls.sinusoidal,
            cls.square_wave,
            cls.sawtooth,
            cls.triangular,
            cls.damped_oscillation,
            cls.gaussian_pulse,
            cls.chirp,
            cls.single_pulse
        ]


# =============================================================================
# 3. DATA GENERATION
# =============================================================================

class SignalDatasetGenerator:
    """Generating noisy and clean signal pairs"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.generators = SignalGenerators.get_all_generators()

    def generate_noisy_signals(
        self,
        n_samples: int,
        noise_eps: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates noisy data for different signal types

Args:
n_samples: Number of samples for each signal type
noise_eps: Noise power (If None, config.epsilon is used)

Returns:
(noisy_signals, clean_signals) - shape: (n_samples * 8 * 5, M)
        """
        if noise_eps is None:
            noise_eps = self.config.epsilon

        M = self.config.M
        t = np.linspace(0, self.config.T, M)

        clean_signals = []
        noisy_signals = []

        # Har bir signal turi uchun
        for generator in self.generators:
            for _ in range(n_samples):
                # Toza signal
                S = generator(t)

                # 5 ta turli shovqinli realizatsiya
                for _ in range(5):
                    # Additive White Gaussian Noise (AWGN)
                    noise = noise_eps * np.random.normal(0, 1, size=M)
                    Y = S + noise

                    clean_signals.append(S)
                    noisy_signals.append(Y)

        return (
            np.array(noisy_signals, dtype=np.float32),
            np.array(clean_signals, dtype=np.float32)
        )

    def generate_single_type(
        self,
        n_samples: int,
        signal_type: int,
        noise_eps: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Generates one type of signal

Args:
n_samples: Number of samples
signal_type: 0-7 (sinusoidal, square, sawtooth, ...)
noise_eps: Noise power

Returns:
(noisy, clean, signal_name)
        """
        if noise_eps is None:
            noise_eps = self.config.epsilon

        M = self.config.M
        t = np.linspace(0, self.config.T, M)

        generators = self.generators
        signal_names = [
            "Sinusoidal", "Square wave", "Sawtooth", "Triangular",
            "Damped oscillation", "Gaussian Pulse", "Chirp", "Single pulse"
        ]

        if signal_type < 0 or signal_type >= len(generators):
            raise ValueError(f"signal_type 0 dan {len(generators)-1} gacha bo'lishi kerak")

        S = generators[signal_type](t)
        name = signal_names[signal_type]

        clean = []
        noisy = []

        for _ in range(n_samples):
            noise = noise_eps * np.random.normal(0, 1, size=M)
            Y = S + noise

            clean.append(S)
            noisy.append(Y)

        return (
            np.array(noisy, dtype=np.float32),
            np.array(clean, dtype=np.float32),
            name
        )


class SignalDataset(Dataset):
    """PyTorch Dataset wrapper"""

    def __init__(self, noisy: np.ndarray, clean: np.ndarray):
        self.noisy = torch.from_numpy(noisy)
        self.clean = torch.from_numpy(clean)

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[idx], self.clean[idx]


# =============================================================================
# 4. SPECTRAL RECONSTRUCTION CORE
# =============================================================================

class SpectralReconstructor(nn.Module):
    """
    Signal reconstruction using spectral basis functions

Trigonometric basis:
        φ_{2k}(t) = √2 * sin(2πkt)
        φ_{2k+1}(t) = √2 * cos(2πkt)
    """

    def __init__(self, max_basis: int, T: float = 1.0):
        super().__init__()
        self.max_basis = max_basis
        self.T = T

        # Vaqt gridini oldindan hisoblash
        self.register_buffer('t_grid', None)
        self.register_buffer('basis_cache', None)

    def _get_phi_matrix(self, j_vals: torch.Tensor, t_vals: torch.Tensor) -> torch.Tensor:
        """
        Creates a trigonometric basis matrix

Args:
j_vals: Basis indices (1, 2, ..., l)
t_vals: Time points [0, T]

Returns:
phi: Basis matrix in shape (l, M)
        """
      
        j_half = torch.floor(j_vals / 2)

      
        angles = 2 * math.pi * torch.tensordot(j_half, t_vals, dims=0)
 
        is_even = ((j_vals % 2) == 0).float().unsqueeze(1)

        sqrt2 = math.sqrt(2.0)
        sin_part = sqrt2 * torch.sin(angles)
        cos_part = sqrt2 * torch.cos(angles)

        phi = is_even * sin_part + (1 - is_even) * cos_part

        ones_col = torch.ones(phi.size(0), 1, device=phi.device)
        # ones_col = torch.ones(phi.size(0), 1, device=phi.device) / math.sqrt(2.0)
        phi = torch.cat([ones_col, phi], dim=1)

        return phi[:, :-1]

    def compute_optimal_l(
        self,
        kappa: torch.Tensor,
        alpha: torch.Tensor,
        r: torch.Tensor,
        epsilon: float,
        max_n: int
    ) -> torch.Tensor:
        """
       Calculates the optimal base length 

Condition: exp(k*l^a) * Sexp(k*m^a) - Sexp(2k*m^a) ≤ r/e² 

Args: 
kappa: parameter k 
alpha: parameter a 
r: The r parameter 
epsilon: Noise level 
max_n: Maximum number of bases 

Returns: 
Optimal l(int)
        """
        device = kappa.device

        # l = 1, 2, ..., max_n
        l_range = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)

        # exp(k * l^α)
        exp_kl_alpha = torch.exp(kappa * (l_range ** alpha))

        sum1 = torch.cumsum(exp_kl_alpha, dim=0)           # Σ exp(k*m^α)
        sum2 = torch.cumsum(torch.exp(2 * kappa * (l_range ** alpha)), dim=0)

        condition = exp_kl_alpha * sum1 - sum2
        threshold = r / (epsilon ** 2)

        mask = (condition <= threshold)

        if mask.any():
            # Shartni qanoatlantiruvchi eng katta l
            return (mask.nonzero(as_tuple=False).max()).int() + 1
        else:
            # Agar hech qaysi l mos kelmasa, minimal qiymat
            return torch.tensor(3, dtype=torch.int32, device=device)

    def forward(
        self,
        Y: torch.Tensor,
        alpha: torch.Tensor,
        kappa: torch.Tensor,
        r: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """
        Signal reconstruction for a single set of parameters

Args:
Y: (batch, M) - input signal
alpha, kappa, r: Scalar parameters
epsilon: Noise level

Returns:
S_hat: (batch, M) - reconstructed signal
        """
        batch_size, M = Y.shape
        device = Y.device

        l = int(self.compute_optimal_l(kappa, alpha, r, epsilon, M).item())
        l = min(l, M - 1)  # Cheklash

        t_vals = torch.linspace(0, self.T, M, device=device)

        j_vals = torch.arange(1, l + 1, dtype=torch.float32, device=device)
        phi_t = self._get_phi_matrix(j_vals, t_vals)  # (l, M)

        omega = torch.matmul(Y, phi_t.T) / M  # (batch, l)

        l_float = float(l)
        j_float = j_vals  # (l,)

        lambda_j = 1.0 - torch.exp(-kappa * ((l_float ** alpha) - (j_float ** alpha)))

        weighted = omega * lambda_j.unsqueeze(0)  # (batch, l)

        S_hat = torch.matmul(weighted, phi_t)  # (batch, M)

        return S_hat


# =============================================================================
# 5. RECURRENT NEURAL NETWORK (BASIC MODEL)
# =============================================================================
iterationsss=[]
class RecurrentSLE(nn.Module):
    """
   Recurrent Signal Learning Engine (SLE-RNN)

At each iteration:
1. Compute S_hat from current Y using spectral method
2. Pass Y_new = S_hat to next iteration (no noise added!)
    """

    def __init__(
        self,
        M: int,
        param_grid: ParameterGrid,
        epsilon: float,
        T: float = 1.0,
        gating_hidden: int = 64
    ):
        """
       Args:
M: Signal length
param_grid: (alpha, kappa, r) ​​parameter grid
epsilon: Noise level
T: Time interval
gating_hidden: Gating network hidden layer size
        """
        super().__init__()

        self.M = M
        self.epsilon = epsilon
        self.T = T
        self.param_grid = param_grid

        self.candidate_params = param_grid.get_all_combinations()
        self.L = len(self.candidate_params)

        print(f"Model yaratildi: {self.L} ta parametrlar kombinatsiyasi")

        self.reconstructor = SpectralReconstructor(M, T)

        self.gating = nn.Sequential(
            nn.Linear(M, gating_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularizatsiya
            nn.Linear(gating_hidden, self.L)
        )

        self._precompute_l_values()

    def _precompute_l_values(self):
        """Pre-calculate the optimal l for all parameters"""
        device = next(self.parameters()).device

        self.l_values = []
        for alpha, kappa, r in self.candidate_params:
            alpha_t = torch.tensor(alpha, device=device)
            kappa_t = torch.tensor(kappa, device=device)
            r_t = torch.tensor(r, device=device)

            l = self.reconstructor.compute_optimal_l(
                kappa_t, alpha_t, r_t, self.epsilon, self.M
            )
            self.l_values.append(l.item())

        self.l_values = torch.tensor(self.l_values, dtype=torch.int32)
        self.max_l = int(self.l_values.max().item())

    def _compute_single_expert(
        self,
        Y: torch.Tensor,
        idx: int
    ) -> torch.Tensor:
        """
        Reconstruction for one expert (parameter combination)

Args:
Y: (batch, M) - input signal
idx: Expert index

Returns:
S_hat: (batch, M) - reconstructed signal
        """
        alpha, kappa, r = self.candidate_params[idx]

        alpha_t = torch.tensor(alpha, device=Y.device, dtype=Y.dtype)
        kappa_t = torch.tensor(kappa, device=Y.device, dtype=Y.dtype)
        r_t = torch.tensor(r, device=Y.device, dtype=Y.dtype)

        return self.reconstructor(Y, alpha_t, kappa_t, r_t, self.epsilon)

    def _iterative_refinement(
        self,
        Y_initial: torch.Tensor,
        num_iterations: int
    ) -> torch.Tensor:
        """
        RECURRENT MAIN PART — With dynamic stopping condition

Iterative signal improvement:
Y^(0) = Y_noisy
for t in 1..T:
S^(t) = Σ w_i * Expert_i(Y^(t-1))
Y^(t) = S^(t) <-- DIFFERENCE: no noise added!

Stopping condition (checked after at least 2 iterations):
MSE(S_hat_new, S_hat_old) <= (kappa^(1/alpha)) * n^(1/alpha) / n
where kappa and alpha are the best expert
parameters selected by gating, n = M (signal length).

Maximum number of iterations: 10

Args:
Y_initial: (batch, M) - initial noisy signal
num_iterations: (ignored, dynamic stopping is used)

Returns:
S_final: (batch, M) - final reconstructed signal
        """
        device = Y_initial.device
        n = float(self.M)  # Signal uzunligi (n)

        MAX_ITERATIONS = 10
        MIN_ITERATIONS = 2


        Y_current = Y_initial
        S_hat_old = None  # Oldingi iteratsiya natijasi
        S_hat = None

        for iteration in range(MAX_ITERATIONS):
            # === 1. GATING: The weight of each expert ===
            logits = self.gating(Y_current)  # (batch, L)
            weights = F.softmax(logits, dim=1)  # (batch, L)

            # === 2. EXPERTS:Reconstruction for each set of parameters ===
            reconstructions = []

            for idx in range(self.L):
                S_hat_i = self._compute_single_expert(Y_current, idx)
                reconstructions.append(S_hat_i)

            reconstructions = torch.stack(reconstructions, dim=1)

            # === 3. AGGREGATION: Weighted Average ===
            weights_expanded = weights.unsqueeze(-1)  # (batch, L, 1)
            S_hat = (reconstructions * weights_expanded).sum(dim=1)

            # === 4. DYNAMIC STOPPING CONDITION ===
            if iteration >= MIN_ITERATIONS - 1 and S_hat_old is not None:
                if iteration<3:
                    best_expert_idx = weights.mean(dim=0).argmax().item()
                    alpha_val, kappa_val, _ = self.candidate_params[best_expert_idx]

                    alpha_safe = max(alpha_val, 1e-6)

                threshold = (kappa_val ** (1.0 / alpha_safe)) * (torch.log(torch.tensor(n)) ** (1.0 / alpha_safe)) / n

                mse_diff = F.mse_loss(S_hat, S_hat_old).item()

                # Debug:print iteration data
                # print(f"Iter {iteration+1}: MSE_diff={mse_diff:.6f}, threshold={threshold:.6f}")

                if mse_diff <= threshold:
                    # print(f"  => Цикл остоновилас: сделано {iteration+1} итерация")
                    iterationsss.append(iteration)
                    break

            # === 5. RECURRENT UPDATE ===
            S_hat_old = S_hat.detach().clone()
            scale = Y_current.norm(dim=-1, keepdim=True) / (S_hat.norm(dim=-1, keepdim=True) + 1e-8)
            Y_current = S_hat * scale
            # Y_current = S_hat

        return  Y_current

    def forward(
        self,
        Y: torch.Tensor,
        num_iterations: Optional[int] = None,
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args: 
Y: (batch, M) - noisy signal(s) 
num_iterations: Maximum iterations (10 if None). 
Dynamic stop condition: 
MSE(S_hat_new, S_hat_old) <= (k^(1/a)) * n^(1/a) / n 
At least 2 iterations are performed. 
return_all: Returns (S_hat, weights) if True 

Returns: 
S_hat: (batch, M) - restored clean signal
        """
        if num_iterations is None:
            num_iterations = 10  # Maksimal chegara (dinamik to'xtatish ishlatiladi)

        # Recurrent iterative improvement (with dynamic stopping condition)
        S_hat = self._iterative_refinement(Y, num_iterations)

        if return_all:
            # Also return the weights for the last iteration
            with torch.no_grad():
                logits = self.gating(S_hat)
                weights = F.softmax(logits, dim=1)
            return S_hat, weights

        return S_hat


# =============================================================================
# 6. PARAMETER SPACE GENERATOR
# =============================================================================

def create_adaptive_param_grid(epsilon: float) -> ParameterGrid:
 
    eps = epsilon

    # Grid o'lchamlari
    m1 = 2 * int((math.log(math.log(1 + (1 / eps**4))))**2)
    m2 = 2 * int((math.log(math.log(1 + (1 / eps**2))))**4)
    m3 = 2 * int((math.log(math.log(1 + (1 / eps**2))))**2)

    print(f"Parametr grid o'lchamlari: α={m1}, κ={m2}, r={m3}")

    # Alpha: [1/m1, 2/m1, ..., 1] / log(log(1/ε⁴))²
    denom_alpha = math.log(math.log(1 + (1 / eps**4)))**2
    alphas = [i / denom_alpha for i in range(1, m1 + 1)]

    # Kappa 
    denom_kappa = math.log(math.log(1 + (1 / eps**2)))**2
    kappas = [i / denom_kappa for i in range(1, m2 + 1)]

    # r
    denom_r = math.log(math.log(1 + (1 / eps**2)))
    rs = [1 + i / denom_r for i in range(1, m3 + 1)]

    return ParameterGrid(alphas=alphas, kappas=kappas, rs=rs)


# =============================================================================
# 7. TRAINING AND ASSESSMENT
# =============================================================================

class Trainer:
    """Model teaching and assessment class"""

    def __init__(
        self,
        model: RecurrentSLE,
        config: ModelConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.criterion = nn.MSELoss()

    def train(
        self,
        train_loader: DataLoader,
        epochs: Optional[int] = None
    ) -> List[float]:
        """
           Train the model

Args:
train_loader: DataLoader
epochs: Number of epochs (config.epochs if None)

Returns:
losses: Average loss per epoch
        """
        if epochs is None:
            epochs = self.config.epochs

        self.model.train()
        history = []

        for epoch in range(epochs):
            total_loss = 0.0
            num_samples = 0

            for batch_noisy, batch_clean in train_loader:
                batch_noisy = batch_noisy.to(self.device)
                batch_clean = batch_clean.to(self.device)

                self.optimizer.zero_grad()

                # Forward: recurrent iterative reconstruction
                S_hat = self.model(
                    batch_noisy,
                    num_iterations=self.config.num_iterations
                )

                loss = self.criterion(S_hat, batch_clean)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_noisy.size(0)
                num_samples += batch_noisy.size(0)

            avg_loss = total_loss / num_samples
            history.append(avg_loss)

            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

        return history

    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Model evaluation

        Returns:
            (mse, predictions, targets)
        """
        self.model.eval()
        total_mse = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_noisy, batch_clean in test_loader:
                batch_noisy = batch_noisy.to(self.device)
                batch_clean = batch_clean.to(self.device)

                S_hat = self.model(
                    batch_noisy,
                    num_iterations=self.config.num_iterations
                )

                mse = F.mse_loss(S_hat, batch_clean, reduction='sum')
                total_mse += mse.item()

                all_preds.append(S_hat.cpu().numpy())
                all_targets.append(batch_clean.cpu().numpy())

        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        avg_mse = total_mse / len(predictions)

        return avg_mse, predictions, targets

    def predict_single(
        self,
        Y: np.ndarray,
        num_iterations: Optional[int] = None
    ) -> np.ndarray:
        """
        Prediction for one signal
        """
        if num_iterations is None:
            num_iterations = self.config.num_iterations

        self.model.eval()

        Y_tensor = torch.from_numpy(Y).unsqueeze(0).to(self.device)

        with torch.no_grad():
            S_hat = self.model(Y_tensor, num_iterations=num_iterations)

        return S_hat.cpu().numpy().squeeze(0)


# =============================================================================
# 8. VISUALIZATION AND ANALYSIS
# =============================================================================

class Visualizer:
    """Visualizing results"""

    @staticmethod
    def plot_training_history(history: List[float], save_path: Optional[str] = None):
        plt.figure(figsize=(10, 6))
        plt.plot(history, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_signal_comparison(
        t: np.ndarray,
        clean: np.ndarray,
        noisy: np.ndarray,
        predicted: np.ndarray,
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        axes[0].plot(t, clean, 'g-', linewidth=2, label='Clean Signal')
        axes[0].set_title('Original Clean Signal')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, noisy, 'r-', alpha=0.7, label='Noisy Signal')
        axes[1].set_title(f'Noisy Signal (SNR: {Visualizer._compute_snr(clean, noisy):.2f} dB)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, clean, 'g--', linewidth=2, alpha=0.7, label='Clean (target)')
        axes[2].plot(t, predicted, 'b-', linewidth=2, label='Predicted')
        axes[2].set_title(f'Reconstructed Signal (MSE: {np.mean((clean-predicted)**2):.6f})')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_multiple_predictions(
        t: np.ndarray,
        predictions: np.ndarray,
        true_signal: np.ndarray,
        n_plot: int = 10
    ):
        """Show multiple predictions"""
        plt.figure(figsize=(12, 6))

        for i in range(min(n_plot, len(predictions))):
            plt.plot(t, predictions[i], 'b-', alpha=0.3, linewidth=1)

        mean_pred = predictions.mean(axis=0)
        plt.plot(t, mean_pred, 'k-', linewidth=3, label='Mean Prediction')

        plt.plot(t, true_signal, 'r--', linewidth=2, label='True Signal')

        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'Signal Predictions (n={len(predictions)})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def _compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
        """Signal-to-Noise Ratio (dB)"""
        signal_power = np.mean(clean**2)
        noise_power = np.mean((noisy - clean)**2)
        return 10 * np.log10(signal_power / noise_power)


# =============================================================================
# 9. MAIN OPERATIONAL FUNCTION
# =============================================================================

def main():
    """MAIN OPERATIONAL FUNCTION"""

    # --- 1. Configuration ---
    config = ModelConfig(
        M=1000,
        epsilon=0.1,
        num_iterations=2,
        epochs=10,
        batch_size=64
        # noise=0.1
    )
    noise=0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Qurilma: {device}")
    print(f"Configuration: {config}")

    # --- 2. Data generation ---
    print("\n--- Data generation ---")
    generator = SignalDatasetGenerator(config)

    # Training information
    Y_train, S_train = generator.generate_noisy_signals(n_samples=15, noise_eps=noise)
    print(f"O'qitish ma'lumotlari: {Y_train.shape}")

    # Test data (one type of signal)
    Y_test, S_test, signal_name = generator.generate_single_type(
        n_samples=20, signal_type=2, noise_eps=noise  # Sawtooth
    )
    print(f"Test data ({signal_name}): {Y_test.shape}")

    # DataLoader yaratish
    train_dataset = SignalDataset(Y_train, S_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    # --- 3. Model creation ---
    print("\n--- Model creation ---")
    param_grid = create_adaptive_param_grid(config.epsilon)

    model = RecurrentSLE(
        M=config.M,
        param_grid=param_grid,
        epsilon=config.epsilon,
        T=config.T,
        gating_hidden=config.gating_hidden
    ).to(device)

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- 4. Train ---
    print("\n--- Train ---")
    trainer = Trainer(model, config, device)
    history = trainer.train(train_loader)

    # --- 5. Visualization ---
    print("\n--- Results ---")
    Visualizer.plot_training_history(history)

    # Test for one signal
    t = np.linspace(0, config.T, config.M)
    pred = trainer.predict_single(Y_test[0])

    Visualizer.plot_signal_comparison(
        t=t,
        clean=S_test[0],
        noisy=Y_test[0],
        predicted=pred
    )

   
    all_preds = [trainer.predict_single(Y_test[i]) for i in range(len(Y_test))]
    all_preds = np.array(all_preds)

    Visualizer.plot_multiple_predictions(t, all_preds, S_test[0])

    # MSE  
    mse = np.mean((all_preds - S_test)**2)
    print(f"\nO'rtacha MSE: {mse:.6f}")

    print("\n--- Finish ---")
    return model, trainer, history


# =============================================================================
# 10. ADDITIONAL FUNCTIONS (FOR ANALYSIS)
# =============================================================================

def analyze_optimal_parameters(
    model: RecurrentSLE,
    generator: SignalDatasetGenerator,
    config: ModelConfig,
    device: torch.device
):
    """
    Analysis of optimal parameters for each signal type
    """
    print("\n=== Optimal Parametrlar Tahlili ===")

    param_stats = []

    for signal_type in range(8):
        Y_test, S_test, name = generator.generate_single_type(
            n_samples=100, signal_type=signal_type, noise_eps=0.01
        )

        Y_tensor = torch.from_numpy(Y_test).to(device)

        model.eval()
        with torch.no_grad():
            _, weights = model(Y_tensor, return_all=True)

        optimal_indices = torch.argmax(weights, dim=1).cpu().numpy()

        alphas, kappas, rs = [], [], []
        for idx in optimal_indices:
            a, k, r = model.candidate_params[idx]
            alphas.append(a)
            kappas.append(k)
            rs.append(r)

        param_stats.append({
            'name': name,
            'alpha_mean': np.mean(alphas),
            'alpha_std': np.std(alphas),
            'kappa_mean': np.mean(kappas),
            'kappa_std': np.std(kappas),
            'r_mean': np.mean(rs),
            'r_std': np.std(rs)
        })

        print(f"\n{name}:")
        print(f"  α = {np.mean(alphas):.4f} ± {np.std(alphas):.4f}")
        print(f"  κ = {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
        print(f"  r = {np.mean(rs):.4f} ± {np.std(rs):.4f}")

    return param_stats


# =============================================================================

if __name__ == "__main__":
    model, trainer, history = main()

    # Additional analysis (optional)
    # config = ModelConfig()
    # generator = SignalDatasetGenerator(config)
    # device = next(model.parameters()).device
    # analyze_optimal_parameters(model, generator, config, device)

# -*- coding: utf-8 -*-
"""
SLE-RNN Model Analysis
========================

This script analyzes the trained RecurrentSLE model and:
1. Determines the optimal parameters for each signal type
2. Visualizes the MoE gating mechanism
3. Shows the level of expert participation
4. Determines "removed" (unused) experts

Usage:
python analyze_model.py --model_path model.pth --output_dir analysis/
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import asdict
import warnings
warnings.filterwarnings('ignore')

 

class ModelAnalyzer:
 

    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()

        # Parametrlar ro'yxati
        self.candidate_params = model.candidate_params
        self.L = len(self.candidate_params)

        # Statistikalar
        self.stats = {}

    def analyze_signal_type(
        self,
        Y_test: np.ndarray,
        S_test: np.ndarray,
        signal_name: str,
        n_samples: Optional[int] = None
    ) -> Dict:
 
        if n_samples is None:
            n_samples = len(Y_test)
        else:
            n_samples = min(n_samples, len(Y_test))

        Y_tensor = torch.from_numpy(Y_test[:n_samples]).to(self.device)

        with torch.no_grad():
            # Forward pass with return_all=True to get weights
            S_hat, weights = self.model(Y_tensor, return_all=True)

            # weights shape: (n_samples, L)
            weights_np = weights.cpu().numpy()

            # The expert with the largest weight for each sample
            dominant_per_sample = np.argmax(weights_np, axis=1)

            # Frequency of use of each expert
            expert_usage = np.bincount(dominant_per_sample, minlength=self.L)
            expert_usage_freq = expert_usage / n_samples  # Normalizatsiya

            # Average values ​​of weights
            mean_weights = weights_np.mean(axis=0)

            # Most active experts (top 5)
            top5_indices = np.argsort(mean_weights)[-5:][::-1]
            dominant_experts = [
                {
                    'rank': i+1,
                    'expert_id': int(idx),
                    'alpha': self.candidate_params[idx][0],
                    'kappa': self.candidate_params[idx][1],
                    'r': self.candidate_params[idx][2],
                    'mean_weight': float(mean_weights[idx]),
                    'usage_frequency': float(expert_usage_freq[idx])
                }
                for i, idx in enumerate(top5_indices)
            ]

            unused_threshold = 0.01 / self.L  
            unused_mask = mean_weights < unused_threshold
            unused_experts = [
                {
                    'expert_id': int(i),
                    'alpha': params[0],
                    'kappa': params[1],
                    'r': params[2],
                    'mean_weight': float(mean_weights[i])
                }
                for i, params in enumerate(self.candidate_params)
                if unused_mask[i]
            ]

 
            eps = 1e-10
            entropy_per_sample = -np.sum(weights_np * np.log(weights_np + eps), axis=1)
            mean_entropy = float(entropy_per_sample.mean())
            max_possible_entropy = np.log(self.L)  # Maksimal 
            normalized_entropy = mean_entropy / max_possible_entropy  # 0-1  

            optimal_alphas = [self.candidate_params[i][0] for i in dominant_per_sample]
            optimal_kappas = [self.candidate_params[i][1] for i in dominant_per_sample]
            optimal_rs = [self.candidate_params[i][2] for i in dominant_per_sample]

            result = {
                'signal_name': signal_name,
                'n_samples': n_samples,
                'optimal_params': {
                    'alpha': {
                        'mean': float(np.mean(optimal_alphas)),
                        'std': float(np.std(optimal_alphas)),
                        'median': float(np.median(optimal_alphas)),
                        'min': float(np.min(optimal_alphas)),
                        'max': float(np.max(optimal_alphas)),
                        'best': float(self.candidate_params[top5_indices[0]][0])
                    },
                    'kappa': {
                        'mean': float(np.mean(optimal_kappas)),
                        'std': float(np.std(optimal_kappas)),
                        'median': float(np.median(optimal_kappas)),
                        'min': float(np.min(optimal_kappas)),
                        'max': float(np.max(optimal_kappas)),
                        'best': float(self.candidate_params[top5_indices[0]][1])
                    },
                    'r': {
                        'mean': float(np.mean(optimal_rs)),
                        'std': float(np.std(optimal_rs)),
                        'median': float(np.median(optimal_rs)),
                        'min': float(np.min(optimal_rs)),
                        'max': float(np.max(optimal_rs)),
                        'best': float(self.candidate_params[top5_indices[0]][2])
                    }
                },
                'expert_usage': {
                    'frequencies': expert_usage_freq.tolist(),
                    'dominant_per_sample': dominant_per_sample.tolist()
                },
                'gating_entropy': {
                    'raw': mean_entropy,
                    'normalized': normalized_entropy,
                    'interpretation': 'Random' if normalized_entropy > 0.8 else
                                    'Partially identified' if normalized_entropy > 0.5 else
                                    'Precisely selected'
                },
                'dominant_experts': dominant_experts,
                'unused_experts': unused_experts,
                'performance': {
                    'mse': float(F.mse_loss(S_hat, torch.from_numpy(S_test[:n_samples]).to(self.device)).item())
                }
            }

            return result

    def analyze_all_signal_types(
        self,
        generator,
        noise_eps: float = 0.1,
        n_samples_per_type: int = 100
    ) -> Dict[str, Dict]:
 
        all_results = {}

        signal_names = [
            "Sinusoidal", "Square wave", "Sawtooth", "Triangular",
            "Damped oscillation", "Gaussian Pulse", "Chirp", "Single pulse"
        ]

        print("=" * 80)
        print("ANALYSIS BY SIGNAL TYPES")
        print("=" * 80)

        for signal_type in range(8):
            print(f"\n--- {signal_names[signal_type]} ({signal_type}) ---")

            Y_test, S_test, name = generator.generate_single_type(
                n_samples=n_samples_per_type,
                signal_type=signal_type,
                noise_eps=noise_eps
            )

            result = self.analyze_signal_type(Y_test, S_test, name, n_samples_per_type)
            all_results[name] = result

            self._print_single_analysis(result)

        self.stats = all_results
        return all_results

    def _print_single_analysis(self, result: Dict):
        p = result['optimal_params']

        print(f"  Optimal α: {p['alpha']['mean']:.4f} ± {p['alpha']['std']:.4f} "
              f"[best: {p['alpha']['best']:.4f}]")
        print(f"  Optimal κ: {p['kappa']['mean']:.4f} ± {p['kappa']['std']:.4f} "
              f"[best: {p['kappa']['best']:.4f}]")
        print(f"  Optimal r: {p['r']['mean']:.4f} ± {p['r']['std']:.4f} "
              f"[best: {p['r']['best']:.4f}]")
        print(f"  Gating entropy: {result['gating_entropy']['normalized']:.3f} "
              f"({result['gating_entropy']['interpretation']})")
        print(f"  MSE: {result['performance']['mse']:.6f}")

        print(f"\n  TOP 5 EXPERTS:")
        for expert in result['dominant_experts']:
            print(f"    #{expert['rank']}. Expert {expert['expert_id']}: "
                  f"α={expert['alpha']:.4f}, κ={expert['kappa']:.4f}, r={expert['r']:.4f} "
                  f"(w={expert['mean_weight']:.4f}, f={expert['usage_frequency']:.2%})")

        if result['unused_experts']:
            print(f"\n  EXPERTS WHO HAVE BEEN REMOVED ({len(result['unused_experts'])} ta):")
            for exp in result['unused_experts'][:5]:  # Faqat birinchi 5 tasini
                print(f"    Expert {exp['expert_id']}: w={exp['mean_weight']:.6f}")
            if len(result['unused_experts']) > 5:
                print(f"     {len(result['unused_experts']) - 5} ta")
        else:
            print(f"\n  All experts are used!")

    def visualize_moe_analysis(self, save_prefix: str = "moe_analysis"):
        """
        MoE tahlili uchun vizualizatsiyalar
        """
        if not self.stats:
            print("Avval analyze_all_signal_types() ni ishga tushiring!")
            return

        # 1. Har bir signal turi uchun ekspert ishlatilishi (heatmap)
        self._plot_expert_usage_heatmap(save_prefix)

        # 2. Dominant ekspertlarning parametrlari (scatter plot)
        self._plot_dominant_experts_scatter(save_prefix)

        # 3. Gating entropiyasi taqqoslash
        self._plot_entropy_comparison(save_prefix)

        # 4. Olib tashlangan ekspertlar
        self._plot_unused_experts(save_prefix)

        # 5. Global ekspert ishlatilishi
        self._plot_global_expert_usage(save_prefix)

    def _plot_expert_usage_heatmap(self, save_prefix: str):
        """Ekspert ishlatilishini heatmap ko'rinishida"""
        signal_names = list(self.stats.keys())

        # Har bir signal uchun ekspert chastotalari
        usage_matrix = np.array([
            self.stats[name]['expert_usage']['frequencies']
            for name in signal_names
        ])  # shape: (8, L)

        # Faqat eng faol 20 ekspertni ko'rsatish (o'qish oson bo'lishi uchun)
        mean_usage = usage_matrix.mean(axis=0)
        top20_indices = np.argsort(mean_usage)[-20:]

        usage_matrix_top20 = usage_matrix[:, top20_indices]

        plt.figure(figsize=(14, 8))
        sns.heatmap(
            usage_matrix_top20,
            xticklabels=[f"E{idx}" for idx in top20_indices],
            yticklabels=signal_names,
            cmap='YlOrRd',
            annot=False,
            cbar_kws={'label': 'Ishlatilish chastotasi'}
        )
        plt.title('Using MoE experts (Top 20)', fontsize=14)
        plt.xlabel('Ekspert ID', fontsize=12)
        plt.ylabel('Signal type', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f"{save_prefix}_expert_usage_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        # print(f"Saved: {save_prefix}_expert_usage_heatmap.png")

    def _plot_dominant_experts_scatter(self, save_prefix: str):
        """Dominant ekspertlarning alpha-kappa-r parametrlarini 3D scatter"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        colors = plt.cm.tab10(np.linspace(0, 1, 8))

        for idx, (name, result) in enumerate(self.stats.items()):
            ax = axes[idx]

            # Dominant ekspertlarning parametrlari
            experts = result['dominant_experts']
            alphas = [e['alpha'] for e in experts]
            kappas = [e['kappa'] for e in experts]
            weights = [e['mean_weight'] for e in experts]

            # Bubble chart: x=alpha, y=kappa, size=weight
            scatter = ax.scatter(
                alphas, kappas,
                s=[w * 5000 for w in weights],  # Size proportional to weight
                c=[weights],
                cmap='viridis',
                alpha=0.6,
                edgecolors='black'
            )

            ax.set_xlabel('α (alpha)', fontsize=10)
            ax.set_ylabel('κ (kappa)', fontsize=10)
            ax.set_title(f'{name}\n(r={experts[0]["r"]:.2f})', fontsize=11)
            ax.grid(True, alpha=0.3)

            # Colorbar
            plt.colorbar(scatter, ax=ax, label='Vazn')

        # plt.suptitle('Dominant Ekspertlarning Parametrlari (Bubble size = Vazn)', fontsize=16)
        plt.tight_layout()
        # plt.savefig(f"{save_prefix}_dominant_experts_scatter.png", dpi=300, bbox_inches='tight')
        plt.show()
        # print(f"Saved: {save_prefix}_dominant_experts_scatter.png")

    def _plot_entropy_comparison(self, save_prefix: str):
        """Gating entropiyasini taqqoslash"""
        signal_names = list(self.stats.keys())
        entropies = [self.stats[name]['gating_entropy']['normalized'] for name in signal_names]

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.barh(signal_names, entropies, color=plt.cm.RdYlGn_r(np.array(entropies)))

        # Qiymatlarni yozish
        for i, (name, val) in enumerate(zip(signal_names, entropies)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalized Entropy', fontsize=12)
        ax.set_title('MoE gate entropy (0=Definite choice, 1=Random)', fontsize=14)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random border')
        ax.legend()

        plt.tight_layout()
        # plt.savefig(f"{save_prefix}_entropy_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        # print(f"Saved: {save_prefix}_entropy_comparison.png")

    def _plot_unused_experts(self, save_prefix: str):
        """Olib tashlangan ekspertlarni ko'rsatish"""
        all_unused = set()
        unused_by_signal = {}

        for name, result in self.stats.items():
            unused_ids = [e['expert_id'] for e in result['unused_experts']]
            unused_by_signal[name] = unused_ids
            all_unused.update(unused_ids)

        if not all_unused:
            print("Olib tashlangan ekspertlar yo'q!")
            return

        # Qaysi signallar uchun qaysi ekspertlar ishlatilmagan
        matrix = np.zeros((len(self.stats), len(all_unused)))
        signal_list = list(self.stats.keys())
        unused_list = sorted(list(all_unused))

        for i, name in enumerate(signal_list):
            for j, exp_id in enumerate(unused_list):
                if exp_id in unused_by_signal[name]:
                    matrix[i, j] = 1

        plt.figure(figsize=(max(10, len(unused_list) * 0.5), 8))
        sns.heatmap(
            matrix,
            xticklabels=[f"E{eid}" for eid in unused_list],
            yticklabels=signal_list,
            cmap='Reds',
            cbar_kws={'label': 'Not used (1) / Used (0)'},
            linewidths=0.5
        )
        plt.title(f'Removed Experts ({len(all_unused)})', fontsize=14)
        plt.xlabel('Ekspert ID', fontsize=12)
        plt.ylabel('Signal type', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f"{save_prefix}_unused_experts.png", dpi=300, bbox_inches='tight')
        plt.show()
        # print(f"Saved: {save_prefix}_unused_experts.png")

        # Statistik ma'lumot
        print(f"\nOlib tashlangan ekspertlar statistikasi:")
        print(f"  Jami olib tashlangan: {len(all_unused)} / {self.L} ({len(all_unused)/self.L:.1%})")
        for name, unused in unused_by_signal.items():
            if unused:
                print(f"  {name}: {len(unused)} ta ekspert olib tashlangan")

    def _plot_global_expert_usage(self, save_prefix: str):
        """Global ekspert ishlatilishi (barcha signallar bo'yicha o'rtacha)"""
        global_usage = np.zeros(self.L)

        for result in self.stats.values():
            global_usage += np.array(result['expert_usage']['frequencies'])

        global_usage /= len(self.stats)  # O'rtacha

        # Saralash
        sorted_indices = np.argsort(global_usage)[::-1]

        plt.figure(figsize=(14, 6))

        # Bar plot
        plt.bar(
            range(self.L),
            global_usage[sorted_indices],
            color=plt.cm.viridis(np.linspace(0, 1, self.L))
        )

        # Eng faol 10 tasini belgilash
        for i in range(min(10, self.L)):
            idx = sorted_indices[i]
            plt.text(
                i, global_usage[idx] + 0.001,
                f'E{idx}\nα={self.candidate_params[idx][0]:.2f}\nκ={self.candidate_params[idx][1]:.2f}',
                ha='center', va='bottom', fontsize=7, rotation=0
            )

        plt.xlabel('Expert (qualified)', fontsize=12)
        plt.ylabel('Average frequency of use', fontsize=12)
        plt.title('Global Expert Advisor Usage (For All Signals)', fontsize=14)
        plt.tight_layout()
        # plt.savefig(f"{save_prefix}_global_expert_usage.png", dpi=300, bbox_inches='tight')
        plt.show()
        # print(f"Saved: {save_prefix}_global_expert_usage.png")

    def generate_report(self, filename: str = "analysis_report.json"):
        """Tahlil natijalarini JSON formatda saqlash"""
        if not self.stats:
            print("Avval tahlilni ishga tushiring!")
            return

        # numpy arraylarni listga aylantirish
        report = {}
        for name, result in self.stats.items():
            report[name] = result

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nTahlil hisoboti saqlandi: {filename}")

    def print_summary_table(self):
        """Xulosa jadvalini chop etish"""
        if not self.stats:
            return

        print("\n" + "=" * 100)
        print("XULOSA JADVALI")
        print("=" * 100)
        print(f"{'Signal':<20} {'α (best)':<10} {'κ (best)':<10} {'r (best)':<10} "
              f"{'Entropiya':<12} {'MSE':<10} {'Unused':<10}")
        print("-" * 100)

        for name, result in self.stats.items():
            p = result['optimal_params']
            n_unused = len(result['unused_experts'])
            print(f"{name:<20} "
                  f"{p['alpha']['best']:<10.4f} "
                  f"{p['kappa']['best']:<10.4f} "
                  f"{p['r']['best']:<10.4f} "
                  f"{result['gating_entropy']['normalized']:<12.3f} "
                  f"{result['performance']['mse']:<10.6f} "
                  f"{n_unused:<10}")


# =============================================================================
# ASOSIY ISHLASH FUNKSIYASI
# =============================================================================

def run_analysis(model, config, generator, device='cpu', output_dir='./analysis'):
    """
    To'liq tahlil ishga tushirish

    Args:
        model: Trained RecurrentSLE model
        config: ModelConfig
        generator: SignalDatasetGenerator
        device: 'cpu' yoki 'cuda'
        output_dir: Natijalar saqlanadigan papka
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Analyzer yaratish
    analyzer = ModelAnalyzer(model, config, device)

    # 1. Barcha signal turlarini tahlil qilish
    print("\n" + "=" * 80)
    print("1. SIGNAL TAHLILI")
    print("=" * 80)
    stats = analyzer.analyze_all_signal_types(generator, noise_eps=0.1, n_samples_per_type=100)

    # 2. Xulosa jadvali
    analyzer.print_summary_table()

    # 3. Vizualizatsiyalar
    print("\n" + "=" * 80)
    print("2. VIZUALIZATSIYALAR")
    print("=" * 80)
    # analyzer.visualize_moe_analysis(save_prefix=f"{output_dir}/moe")

    # 4. Hisobot saqlash
    # analyzer.generate_report(f"{output_dir}/analysis_report.json")

    print("\n" + "=" * 80)
    print("3. ASOSIY XULOSLAR")
    print("=" * 80)

    # Eng yaxshi parametrlar
    print("\nEng ko'p ishlatilgan ekspertlar:")
    global_usage = np.zeros(analyzer.L)
    for result in stats.values():
        global_usage += np.array(result['expert_usage']['frequencies'])
    global_usage /= len(stats)

    top5_global = np.argsort(global_usage)[-5:][::-1]
    for rank, idx in enumerate(top5_global, 1):
        alpha, kappa, r = analyzer.candidate_params[idx]
        print(f"  #{rank}. Expert {idx}: α={alpha:.4f}, κ={kappa:.4f}, r={r:.4f} "
              f"(f={global_usage[idx]:.2%})")

    # Umumiy olib tashlanganlar
    all_unused = set()
    for result in stats.values():
        all_unused.update([e['expert_id'] for e in result['unused_experts']])

    print(f"\nUmumiy olib tashlangan ekspertlar: {len(all_unused)} / {analyzer.L} "
          f"({len(all_unused)/analyzer.L:.1%})")

    return analyzer


# =============================================================================
# MISOL ISHLATISH (main)
# =============================================================================

if __name__ == "__main__":
    """
    MISOL: Model allaqachon trained bo'lsa, quyidagicha ishlatiladi:

    # Model yuklash
    model = RecurrentSLE(...)
    model.load_state_dict(torch.load('model.pth'))

    # Generator yaratish
    config = ModelConfig(M=100, epsilon=0.01)
    generator = SignalDatasetGenerator(config)

    # Tahlil
    analyzer = run_analysis(model, config, generator, device='cpu')
    """
    # MISOL: Model allaqachon trained bo'lsa, quyidagicha ishlatiladi:

    # Model yuklash
    model = model
    # model.load_state_dict(torch.load('model.pth'))

    # Generator yaratish
    config = ModelConfig(M=1000, epsilon=0.1)
    generator = SignalDatasetGenerator(config)

    # Tahlil
    analyzer = run_analysis(model, config, generator, device='cpu')


    print("""
    ============================================================
    MODEL TAHLILI SKRIPTI
    ============================================================

    Bu skript trained RecurrentSLE modelini tahlil qiladi.

    Foydalanish:
        1. Modelingizni yuklang
        2. ModelAnalyzer class'ini yarating
        3. analyze_all_signal_types() ni ishga tushiring
        4. visualize_moe_analysis() ni chaqiring

    Misol kod:
        model = RecurrentSLE(...)  # Sizning modelingiz
        model.load_state_dict(torch.load('model.pth'))

        analyzer = ModelAnalyzer(model, config, device)
        analyzer.analyze_all_signal_types(generator)
        analyzer.visualize_moe_analysis()
        analyzer.print_summary_table()
    """)

print("\n--- Сравнение предсказаний для разных сигналов и уровней шума ---")

def plot_multi_signal_noise_comparison(
    t: np.ndarray,
    results: List[Dict], # List of {'signal_type_name', 'noise_eps', 'clean', 'noisy', 'predicted'}
    title: str = 'Model Predictions Across Different Signals and Noise Levels',
    save_path: Optional[str] = None
):
    unique_signal_names = sorted(list(set([r['signal_type_name'] for r in results])))
    unique_noise_eps = sorted(list(set([r['noise_eps'] for r in results])))

    num_signals = len(unique_signal_names)
    num_noise_levels = len(unique_noise_eps)

    fig, axes = plt.subplots(
        num_signals, num_noise_levels,
        figsize=(num_noise_levels * 6, num_signals * 4),
        sharex=True, sharey=True
    )

    if num_signals == 1 and num_noise_levels == 1:
        axes = np.array([[axes]]) # Ensure axes is 2D for consistent indexing
    elif num_signals == 1:
        axes = np.expand_dims(axes, axis=0) # Ensure axes is 2D
    elif num_noise_levels == 1:
        axes = np.expand_dims(axes, axis=1) # Ensure axes is 2D

    for i, signal_name in enumerate(unique_signal_names):
        for j, noise_eps in enumerate(unique_noise_eps):
            ax = axes[i, j]
            current_result = next(
                (r for r in results if r['signal_type_name'] == signal_name and r['noise_eps'] == noise_eps),
                None
            )

            if current_result:
                clean = current_result['clean']
                noisy = current_result['noisy']
                predicted = current_result['predicted']

                ax.plot(t, clean, 'g--', label='Clean Signal', alpha=0.8, linewidth=2)
                ax.plot(t, noisy, 'r-', label='Noisy Signal', alpha=0.5)
                ax.plot(t, predicted, 'b-', label='Predicted Signal', alpha=0.8)

                if i == 0: # Title for columns (noise levels)
                    ax.set_title(f'Noise: {noise_eps}', fontsize=12)

                if j == 0: # Label for rows (signal types)
                    ax.text(-0.1, 0.5, signal_name, rotation=90, va='center', ha='right', transform=ax.transAxes, fontsize=12)

                if i == num_signals - 1: # X-label for bottom row
                    ax.set_xlabel('Time')

                # Calculate MSE for title
                mse = np.mean((clean - predicted)**2)
                ax.text(0.05, 0.9, f'MSE: {mse:.4f}', transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))

                ax.grid(True, alpha=0.3)

    # Add a single legend outside all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=10)

    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.97]) # Adjust layout to make space for suptitle and legend

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# --- Параметры для сравнения ---
# Выбираем несколько типов сигналов (индексы из SignalGenerators.get_all_generators())
signal_types_to_compare = [
    0, # Sinusoidal
    1, # Square wave
    2,3,4,5,6,7  # Gaussian Pulse
]
signal_names_map = [
    "Sinusoidal", "Square wave", "Sawtooth", "Triangular",
    "Damped oscillation", "Gaussian Pulse", "Chirp", "Single pulse"
]

# Выбираем несколько уровней шума
noise_levels_to_compare = [0.5]

# Время для сигнала
t = np.linspace(0, config.T, 1000)

comparison_results = []

print(f"Начало генерации данных для {len(signal_types_to_compare)} сигналов и {len(noise_levels_to_compare)} уровней шума...")

for signal_type_idx in signal_types_to_compare:
    current_signal_name = signal_names_map[signal_type_idx]
    for noise_eps in noise_levels_to_compare:
        print(f"  Генерация: {current_signal_name}, Epsilon={noise_eps}")

        # Генерируем тестовый сигнал с текущим уровнем шума
        Y_test_single, S_test_single, _ = generator.generate_single_type(
            n_samples=1, # Только один образец
            signal_type=signal_type_idx,
            noise_eps=noise_eps
        )

        # Предсказываем восстановленный сигнал
        predicted_signal = trainer.predict_single(Y_test_single[0])

        comparison_results.append({
            'signal_type_name': current_signal_name,
            'noise_eps': noise_eps,
            'clean': S_test_single[0],
            'noisy': Y_test_single[0],
            'predicted': predicted_signal
        })

print("Данные сгенерированы. Построение графика...")

# Построение общего графика
plot_multi_signal_noise_comparison(
    t=t,
    results=comparison_results,
    title='Сравнение предсказаний модели для разных сигналов и уровней шума',
    save_path='./analysis/multi_signal_noise_comparison.png'
)

print("График сравнения построен и сохранен в ./analysis/multi_signal_noise_comparison.png")

import pandas as pd
import numpy as np

# Define the epsilon values requested by the user
epsilon_values = [0.05, 0.1, 0.3, 0.5, 1.0]

# Prepare to store results
results_table = []

print("\n--- Har bir signal turi bo'yicha epsilon uchun MSE tahlili ---\n")

# Iterate through each of the 8 signal types
for signal_type_idx in range(8):
    current_signal_name = signal_names_map[signal_type_idx]
    row_data = {'Signal turi': current_signal_name}

    print(f"Signal turi: {current_signal_name}")

    # Generate the *clean* signal once for the current signal type
    # We generate it with noise_eps=0 to ensure we get the pure clean signal.
    # generator.generate_single_type returns (noisy, clean, name)
    # Since n_samples=1, S_test_base_single_sample_array will be an array of shape (1, M)
    _, S_test_base_single_sample_array, _ = generator.generate_single_type(
        n_samples=1,
        signal_type=signal_type_idx,
        noise_eps=0 # Generate a clean signal
    )
    S_clean_signal = S_test_base_single_sample_array[0] # Extract the single clean signal (shape M,)

    # Iterate through each epsilon value to add different noise levels to the SAME clean signal
    for eps in epsilon_values:
        # Create a *new noisy version* of the *same clean signal* for the current epsilon
        M = config.M # Get signal length from config
        # Ensure noise is generated as float32 to match model's expected dtype
        noise = eps * np.random.normal(0, 1, size=M).astype(np.float32)
        Y_noisy_for_current_eps = S_clean_signal + noise

        # Predict the reconstructed signal from this specific noisy version
        # trainer.predict_single will convert this numpy array to a torch.tensor
        # which will then have the correct dtype (float32)
        predicted_signal = trainer.predict_single(Y_noisy_for_current_eps)

        # Calculate Mean Squared Error (MSE) using the *original clean signal*
        mse = np.mean((predicted_signal - S_clean_signal)**2)
        row_data[f'Epsilon={eps}'] = f'{mse:.6f}'

    results_table.append(row_data)

# Create a pandas DataFrame for better table formatting
df_mse = pd.DataFrame(results_table)

print("\n--- MSE natijalari jadvali ---\n")
print(df_mse.to_markdown(index=False))

print("\nTahlil yakunlandi.")

print("\n--- Turli epsilon (shovqin) darajalarida signalni taqqoslash ---")

# Namoyish uchun turli epsilon qiymatlari
noise_levels_for_display = [0.01, 0.05, 0.1, 0.3, 0.5,1]

# Test uchun signal turini tanlash (masalan, 0 - Sinusoidal signal)
signal_type_to_test = 7
selected_signal_name = signal_names_map[signal_type_to_test]

# Signal vaqti uchun grid
t = np.linspace(0, config.T, config.M)

print(f"\nTanlangan signal turi: {selected_signal_name}\n")

# Asl toza signalni bir marta generatsiya qilish (shovqinsiz)
_, S_clean_single_array, _ = generator.generate_single_type(
    n_samples=1,
    signal_type=signal_type_to_test,
    noise_eps=0 # Toza signalni olish uchun
)
S_clean_signal = S_clean_single_array[0]

# Matplotlib uslubini va shrift o'lchamlarini sozlash
plt.style.use('seaborn-v0_8-whitegrid') # Yaxshi dizayn uslubini qo'llash
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.titlesize': 22 # Umumiy sarlavha uchun
})

for noise_eps in noise_levels_for_display:
    print(f"Тестирование с epsilon = {noise_eps}")

    # Asl toza signalga joriy epsilon darajasida shovqin qo'shish
    noise = noise_eps * np.random.normal(0, 1, size=config.M).astype(np.float32)
    Y_noisy_for_display = S_clean_signal + noise

    # Qayta tiklangan signalni bashorat qilish
    predicted_signal = trainer.predict_single(Y_noisy_for_display)

    # Natijalarni bitta grafikda vizualizatsiya qilish
    plt.figure(figsize=(12, 6))
    plt.plot(t, Y_noisy_for_display, 'r-', alpha=0.7, label='Noisy Signal')
    plt.plot(t, S_clean_signal, 'g--', linewidth=2, label='Original Signal')
    plt.plot(t, predicted_signal, 'b-', linewidth=2, label='Predicted Signal')

    # plt.title(f'Signalni taqqoslash (Epsilon = {noise_eps}, Turi: {selected_signal_name})')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.6) # Grid chiziqlarini yanada ko'rinadigan qilish
    plt.tight_layout()
    plt.show()

print("\nVizualizatsiya yakunlandi.")

# Assuming config, trainer, generator, t, signal_names_map are available from previous cells.
# Ensure all necessary imports are present for this cell, though they should be from AdtHgfGm6fBu
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

# --- 1. Define new combined signal generation functions ---

def generate_sinus_square(t: np.ndarray) -> np.ndarray:
    return SignalGenerators.sinusoidal(t) + SignalGenerators.square_wave(t)

def generate_sinus_chirp(t: np.ndarray) -> np.ndarray:
    return SignalGenerators.sinusoidal(t) + SignalGenerators.chirp(t)

def generate_square_damp(t: np.ndarray) -> np.ndarray:
    return SignalGenerators.square_wave(t) + SignalGenerators.damped_oscillation(t)

def generate_sinus_half_square_half(t: np.ndarray, M: int) -> np.ndarray:
    mid_point = M // 2
    t_first_half = t[:mid_point]
    t_second_half = t[mid_point:]
    signal_first_half = SignalGenerators.sinusoidal(t_first_half)
    signal_second_half = SignalGenerators.square_wave(t_second_half)
    return np.concatenate((signal_first_half, signal_second_half)).astype(np.float32)

def generate_square_half_damped_half(t: np.ndarray, M: int) -> np.ndarray:
    mid_point = M // 2
    t_first_half = t[:mid_point]
    t_second_half = t[mid_point:]
    signal_first_half = SignalGenerators.square_wave(t_first_half)
    signal_second_half = SignalGenerators.damped_oscillation(t_second_half)
    return np.concatenate((signal_first_half, signal_second_half)).astype(np.float32)

def generate_sinus_half_single_half(t: np.ndarray, M: int) -> np.ndarray:
    mid_point = M // 2
    t_first_half = t[:mid_point]
    t_second_half = t[mid_point:]
    signal_first_half = SignalGenerators.sinusoidal(t_first_half)
    signal_second_half = SignalGenerators.single_pulse(t_second_half)
    return np.concatenate((signal_first_half, signal_second_half)).astype(np.float32)

# --- 2. List of new combined signals and their names ---
combined_signal_generators = {
    "Sinus + Square": lambda t_arr: generate_sinus_square(t_arr),
    "Sinus + Chirp": lambda t_arr: generate_sinus_chirp(t_arr),
    "Square + Damped": lambda t_arr: generate_square_damp(t_arr),
    "Sinus[:500] + Square[500:]": lambda t_arr: generate_sinus_half_square_half(t_arr, config.M),
    "Square[:500] + Damped[500:]": lambda t_arr: generate_square_half_damped_half(t_arr, config.M),
    "Sinus[:500] + Single[500:]": lambda t_arr: generate_sinus_half_single_half(t_arr, config.M)
}

# --- 3. Define epsilon values for analysis ---
epsilon_values = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

# --- 4. Prepare time array ---
M = config.M
t = np.linspace(0, config.T, M)

# --- 5. Iterate through combined signals, calculate MSE, and plot ---
print("--- Yangi kombinatsiyalangan signallarni tahlil qilish ---")

# Matplotlib uslubini va shrift o'lchamlarini sozlash (taking from nt5QTY6mNSXu to ensure consistent style)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.titlesize': 22
})

results_table_combined_signals = []

for signal_name, signal_func in combined_signal_generators.items():
    print(f"\n--- {signal_name} signalini tahlil qilish ---\n")

    # Generate the clean combined signal ONCE
    S_clean_combined = signal_func(t).astype(np.float32)

    row_data = {'Signal turi': signal_name}

    for eps in epsilon_values:
        print(f"  Epsilon = {eps} da baholash...")

        # Add noise to the *same* clean combined signal for current epsilon
        noise = eps * np.random.normal(0, 1, size=M).astype(np.float32)
        Y_noisy_combined = S_clean_combined + noise

        # Predict the reconstructed signal
        predicted_signal_combined = trainer.predict_single(Y_noisy_combined)

        # Calculate MSE
        mse_combined = np.mean((predicted_signal_combined - S_clean_combined)**2)
        print(f"    MSE: {mse_combined:.6f}")
        row_data[f'Epsilon={eps}'] = f'{mse_combined:.6f}'

        # Plotting for each epsilon
        plt.figure(figsize=(14, 7))
        plt.plot(t, Y_noisy_combined, 'r-', alpha=0.7, label='Noisy Signal')
        plt.plot(t, S_clean_combined, 'g--', linewidth=2, label='Original Signal')
        plt.plot(t, predicted_signal_combined, 'b-', linewidth=2, label='Predicted Signal')

        # plt.title(f'{signal_name} uchun signalni taqqoslash (\varepsilon = {eps}, MSE: {mse_combined:.6f})')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.6)
        plt.tight_layout()
        plt.show()

    results_table_combined_signals.append(row_data)

print("\n--- Barcha yangi kombinatsiyalangan signallar tahlili yakunlandi ---")

df_mse_combined = pd.DataFrame(results_table_combined_signals)

print("\n### Kombinatsiyalangan Signallar Uchun MSE Natijalari Jadvali\n")
print(df_mse_combined.to_markdown(index=False))



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# 1. Donoho-Johnstone test signallarini yaratish funksiyalari
# =============================================================================

def blocks(t: np.ndarray) -> np.ndarray:
    """Blocks signal."""
    pos = np.array([.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
    hgt = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
    signal = np.zeros_like(t, dtype=np.float32)
    # Fix: Apply 'h' cumulatively for t >= p to create a step function.
    for p, h in zip(pos, hgt):
        signal[t >= p] += h
    return signal

def bumps(t: np.ndarray) -> np.ndarray:
    """Bumps signal."""
    loc = np.array([.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
    hgt = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
    wth = np.array([.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005])
    signal = np.zeros_like(t, dtype=np.float32)
    for l, h, w in zip(loc, hgt, wth):
        signal += h / ((1 + np.abs(t - l) / w)**4)
    return signal

def heavysine(t: np.ndarray) -> np.ndarray:
    """Heavysine signal."""
    return (4 * np.sin(4 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t)).astype(np.float32)

def doppler(t: np.ndarray) -> np.ndarray:
    """Doppler signal."""
    return np.sqrt(t * (1 - t)) * np.sin(2 * np.pi * (2.1 / (t + 0.05))).astype(np.float32)

# =============================================================================
# 2. Tahlilni amalga oshirish
# =============================================================================

print("\n--- Donoho–Johnstone test signallarini tahlil qilish ---")

dj_signals = {
    "Blocks": blocks,
    "Bumps": bumps,
    "Heavysine": heavysine,
    "Doppler": doppler
}

epsilon_values = [ 0.05, 0.1, 0.3, 0.5,1] # Test uchun epsilon qiymatlari
M = config.M # Signal uzunligi
t = np.linspace(0, config.T, M) # Vaqt vektori

results_mse = []

# Matplotlib uslubini va shrift o'lchamlarini sozlash
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.titlesize': 22
})

for signal_name, signal_func in dj_signals.items():
    # if signal_name != "Doppler": # Faqat Doppler signalini tahlil qilish
    #     continue

    print(f"\nSignal turi: {signal_name}")
    row_data = {"Signal turi": signal_name}

    clean_signal_raw = signal_func(t) # Toza signalni generatsiya qilish (original amplitudada)
    # The following print statements are for debugging and can be removed for cleaner output
    # print(f"{signal_name} raw signal (first 10): {clean_signal_raw[:10]}")
    # print(f"{signal_name} raw signal (min, max): {clean_signal_raw.min()}, {clean_signal_raw.max()}")

    max_amplitude = np.max(np.abs(clean_signal_raw))
    # print(f"{signal_name} max_amplitude: {max_amplitude}")

    if max_amplitude > 1e-6: # Kichik son bilan taqqoslash, 0 ga bo'lishdan saqlanish
        clean_signal = clean_signal_raw / max_amplitude # [-1, 1] oralig'iga normallashtirish
    else:
        clean_signal = clean_signal_raw # Agar signal nol bo'lsa, o'zgarishsiz qoldirish
    # print(f"{signal_name} normalized signal (first 10): {clean_signal[:10]}")
    # print(f"{signal_name} normalized signal (min, max): {clean_signal.min()}, {clean_signal.max()}")

    for eps in epsilon_values:
        # Shovqin qo'shish
        noise = eps * np.random.normal(0, 1, M).astype(np.float32)
        # Ensure noisy_signal is float32 to avoid dtype mismatch with model
        noisy_signal = (clean_signal + noise).astype(np.float32)

        # Model orqali shovqinni tozalash
        predicted_signal = trainer.predict_single(noisy_signal)

        # MSE hisoblash
        mse = np.mean((predicted_signal - clean_signal)**2)
        row_data[f'Epsilon={eps}'] = f'{mse:.6f}'
        print(f"  Epsilon={eps:.2f}, MSE={mse:.6f}")

        # Natijalarni grafikda ko'rsatish (vaqtincha o'chirilgan)
        plt.figure(figsize=(14, 7))
        plt.plot(t, noisy_signal, 'r-', alpha=0.7, label='Shovqinli Signal')
        plt.plot(t, clean_signal, 'g--', linewidth=2, label='Toza Signal')
        plt.plot(t, predicted_signal, 'b-', linewidth=2, label='Model Tiklagan Signal')
        plt.title(f'{signal_name} signalini shovqinlardan tozalash (Epsilon = {eps:.2f}, MSE = {mse:.6f})')
        plt.xlabel('Vaqt')
        plt.ylabel('Amplituda')
        plt.legend()
        plt.grid(True, alpha=0.6)
        plt.tight_layout()
        plt.show()

    results_mse.append(row_data)

# MSE natijalarini jadval shaklida chiqarish
df_dj_mse = pd.DataFrame(results_mse)
print("\n--- Donoho–Johnstone signallari uchun MSE natijalari jadvali ---")
print(df_dj_mse.to_markdown(index=False))

print("\n--- Donoho–Johnstone tahlili yakunlandi ---")

