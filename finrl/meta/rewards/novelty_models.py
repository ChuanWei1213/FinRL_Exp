import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from typing import Dict, Any, Optional
import pickle
import logging
import warnings

# logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')
class SurpriseModel:
    """驚喜模型實現"""
    def __init__(self, observation_dim: int, action_dim: int, device: torch.device):
        self.device = device
        
        # 預測器網絡
        self.predictor = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        ).to(device)
        
        # 均值和標準差預測
        self.mean_head = nn.Linear(64, observation_dim).to(device)
        self.std_head = nn.Sequential(
            nn.Linear(64, observation_dim),
            nn.Softplus()
        ).to(device)
        
        self.optimizer = optim.Adam(
            list(self.predictor.parameters()) + 
            list(self.mean_head.parameters()) + 
            list(self.std_head.parameters()),
            lr=0.001
        )
    
    def compute_bonus(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
        """計算驚喜獎勵"""
        with torch.no_grad():
            x = torch.FloatTensor(np.concatenate([obs, action])).to(self.device)
            next_obs = torch.FloatTensor(next_obs).to(self.device)
            
            features = self.predictor(x)
            mean = self.mean_head(features)
            std = self.std_head(features) + 1e-6
            
            dist = Normal(mean, std)
            log_prob = dist.log_prob(next_obs).sum()
            return -log_prob.item()
    
    def train_step(self, replay_pool, batch_size: int) -> float:
        """訓練動態模型"""
        batch = replay_pool.random_batch(batch_size)
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        next_obs = torch.FloatTensor(batch['next_observations']).to(self.device)
        
        x = torch.cat([obs, actions], dim=1)
        features = self.predictor(x)
        mean = self.mean_head(features)
        std = self.std_head(features) + 1e-6
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(next_obs).sum(dim=1)
        loss = -log_prob.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class DejavuModel:
    """既視感模型實現"""
    def __init__(self, observation_dim: int, device: torch.device, latent_dim: int = 32):
        self.device = device
        
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        ).to(device)
        
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, observation_dim)
        ).to(device)
        
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()),
            lr=0.001
        )
    
    def compute_bonus(self, obs: np.ndarray) -> float:
        """計算既視感獎勵"""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            latent = self.encoder(obs)
            recon = self.decoder(latent)
            recon_error = F.mse_loss(recon, obs, reduction='sum')
            return recon_error.item()
    
    def train_step(self, replay_pool, batch_size: int) -> float:
        """訓練自動編碼器"""
        batch = replay_pool.random_batch(batch_size)
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        
        latent = self.encoder(obs)
        recon = self.decoder(latent)
        loss = F.mse_loss(recon, obs)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item() 
    
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class GaussianMLPRegressor:
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_sizes=(32, 32),
        init_std=1.0,
        normalize_inputs=True,
        normalize_outputs=True,
        weight_decay=0.01,
        learning_rate=0.01,
    ):
        """
        A Gaussian MLP Regressor for regression tasks.
        Args:
            input_dim (int): Dimensionality of input.
            output_dim (int): Dimensionality of output.
            hidden_sizes (tuple): Hidden layer sizes for the mean network.
            init_std (float): Initial standard deviation for output.
            normalize_inputs (bool): Whether to normalize inputs.
            normalize_outputs (bool): Whether to normalize outputs.
            weight_decay (float): Weight decay for L2 regularization.
            learning_rate (float): Learning rate for optimizer.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.init_std = init_std
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Normalization variables
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        # Build model
        self.model = self._build_model().to(self.device)
        # self.optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # self.train_loss_list = []                           

    def _build_model(self):
        """
        Build the MLP model for mean and log-std estimation.
        """
        layers = []
        prev_dim = self.input_dim
        for size in self.hidden_sizes:
            linear = nn.Linear(prev_dim, size)
            # 使用更好的初始化方法
            nn.init.kaiming_normal_(linear.weight)
            nn.init.constant_(linear.bias, 0)
            
            layers.append(linear)
            layers.append(nn.ReLU())
            prev_dim = size
        layers.append(nn.Linear(prev_dim, self.output_dim))  # Mean output layer

        mean_net = nn.Sequential(*layers)

        # Log-std is a learnable parameter
        log_std = nn.Parameter(torch.ones(self.output_dim) * torch.log(torch.tensor(self.init_std)))

        # Wrap the networks into a single module
        class MLPModel(nn.Module):
            def __init__(self, mean_net, log_std):
                super().__init__()
                self.mean_net = mean_net
                self.log_std = log_std

            def forward(self, x):
                mean = self.mean_net(x)
                log_std = self.log_std.expand_as(mean)
                return mean, log_std

        return MLPModel(mean_net, log_std)

    def _normalize(self, x, mean, std):
        """归一化输入"""
        # 确保在同一设备上
        device = x.device
        mean = mean.to(device)
        std = std.to(device)
        return (x - mean) / std

    def _denormalize(self, x, mean, std):
        return x * std + mean

    def _log_likelihood(self, y, mean, log_std):
        """
        Calculate the log-likelihood of y given mean and log_std.
        """
        std = torch.exp(log_std)
        constant = torch.log(torch.tensor(2 * torch.pi))
        # print(f'log_std: {log_std}')
        # import ipdb; ipdb.set_trace()
        log_likelihood = -0.5 * torch.sum(constant + log_std + ((y - mean))**2/ std, dim=-1)
        # log_likelihood = -0.5 * torch.sum(((y - mean) / std)**2 + 2 * log_std + constant, dim=-1)
        # print(f"log_std: {log_std}")
        # print(f"std: {std}")
        # print(f"mean: {mean}")
        # print(f"log_likelihood: {log_likelihood}")
        return log_likelihood

    def _compute_loss(self, x, y):
        """
        Compute the loss (negative log-likelihood + L2 regularization).
        """
        mean, log_std = self.model(x)
        # print(f"x: {x}")
        nll = -self._log_likelihood(y, mean, log_std).mean()
        l2_penalty = self.weight_decay * sum(torch.sum(param**2) for param in self.model.parameters())
        # print(f"nll: {nll}")
        # print(f"l2_penalty: {l2_penalty}")
        return nll + l2_penalty

    def fit(self, x_train, y_train, epochs=10):
        """
        Fit the model using (x_train, y_train).
        """
        if self.normalize_inputs:
            self.x_mean = torch.mean(x_train, dim=0, keepdim=True)
            self.x_std = torch.std(x_train, dim=0, keepdim=True) + 1e-8
            x_train = self._normalize(x_train, self.x_mean, self.x_std)

        if self.normalize_outputs:
            self.y_mean = torch.mean(y_train, dim=0, keepdim=True)
            self.y_std = torch.std(y_train, dim=0, keepdim=True) + 1e-8
            y_train = self._normalize(y_train, self.y_mean, self.y_std)

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        

        self.model.train()
        for epoch in range(epochs):
            def closure():
                self.optimizer.zero_grad()
                loss = self._compute_loss(x_train, y_train)
                loss.backward()
                return loss

            self.optimizer.step(closure)

            # Monitor loss
            with torch.no_grad():
                loss = self._compute_loss(x_train, y_train)
                # if loss is nan, break
                if torch.isnan(loss):
                    print("Loss is nan, stopping training")
                    print(x_train)
                    print(y_train)
                    break
                # self.train_loss_list.append(loss.item())
            # print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        return loss.item()

    def plot_train_loss(self):
        plt.plot(self.train_loss_list)
        plt.show()

    def predict(self, x):
        """
        Predict mean and log-std for given x.
        """

        if self.x_mean is None or self.x_std is None:
            raise ValueError("x_mean or x_std is None, please fit the model first")
        
        if self.normalize_inputs:
            x = self._normalize(x, self.x_mean, self.x_std)

        self.model.eval()
        with torch.no_grad():
            mean, log_std = self.model(x)

        if self.normalize_outputs:
            mean = self._denormalize(mean, self.y_mean, self.y_std)

        return mean, log_std

    def predict_log_likelihood(self, x, y):
        """预测给定输入和输出的对数似然"""
        if self.x_mean is None or self.x_std is None:
            warnings.warn("x_mean or x_std is None, please fit the model first")
            return torch.zeros(1).to(x.device)
        
        # 确保所有张量在同一设备上
        device = next(self.model.parameters()).device
        x = x.to(device)
        y = y.to(device)
        
        # 确保 x_mean 和 x_std 在正确的设备上
        x_mean = self.x_mean.to(device)
        x_std = self.x_std.to(device)
        
        # 归一化输入
        x = self._normalize(x, x_mean, x_std)
        
        # 前向传播预测 - 模型返回均值和对数标准差的元组
        self.model.eval()
        with torch.no_grad():
            mean, log_std = self.model(x)  # 正确解包元组
        
        # 如果需要归一化输出，确保形状正确匹配
        if self.normalize_outputs and self.y_mean is not None and self.y_std is not None:
            y_mean = self.y_mean.to(device)
            y_std = self.y_std.to(device)
            
            # 确保形状兼容
            if mean.dim() > y_mean.dim():
                y_mean = y_mean.view(*([1] * (mean.dim() - y_mean.dim())), *y_mean.shape)
                y_std = y_std.view(*([1] * (mean.dim() - y_std.dim())), *y_std.shape)
            
            # 反归一化预测的均值
            mean = self._denormalize(mean, y_mean, y_std)
            
            # 归一化目标值
            y_normalized = self._normalize(y, y_mean, y_std)
        else:
            y_normalized = y
        
        # 使用高斯分布的对数似然公式计算
        std = torch.exp(log_std)
        var = std.pow(2)
        
        # 确保形状匹配
        if y_normalized.shape != mean.shape:
            # 如果只是批次维度不同，尝试广播
            if y_normalized.numel() == mean.numel():
                y_normalized = y_normalized.view_as(mean)
            else:
                # 取最小共同维度
                min_dim = min(y_normalized.shape[-1], mean.shape[-1])
                y_normalized = y_normalized[..., :min_dim]
                mean = mean[..., :min_dim]
                std = std[..., :min_dim]
                var = var[..., :min_dim]
        
        # 计算对数似然
        log_likelihood = -0.5 * ((y_normalized - mean).pow(2) / var + log_std * 2 + np.log(2 * np.pi)).sum(dim=-1)
        
        return log_likelihood
    
    def compute_bonus(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
        # import ipdb; ipdb.set_trace()

        device = next(self.model.parameters()).device
    
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        if not isinstance(next_obs, torch.Tensor):
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
        
        obs = obs.to(device)
        action = action.to(device)
        next_obs = next_obs.to(device)
        
        x = torch.cat([obs, action.reshape(-1)])
        y = next_obs
        log_likelihood = self.predict_log_likelihood(x, y)
        # import ipdb; ipdb.set_trace()
        surprise = -log_likelihood
        return surprise
    
    def train_step(self, replay_pool, batch_size: int) -> float:
        """訓練動態模型"""
        batch = replay_pool.random_batch(batch_size)
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        next_obs = torch.FloatTensor(batch['next_observations']).to(self.device)
        
        x = torch.cat([obs, actions], dim=1).to(self.device)
        y = next_obs.to(self.device)

        loss = self.fit(x, y, epochs=1)
        return loss
    
    def to(self, device):
        """将网络移动到指定设备"""
        self.device = device
        self.model = self.model.to(device)
        if hasattr(self, 'x_mean') and self.x_mean is not None:
            self.x_mean = self.x_mean.to(device)
        if hasattr(self, 'x_std') and self.x_std is not None:
            self.x_std = self.x_std.to(device)
        if hasattr(self, 'y_mean') and self.y_mean is not None:
            self.y_mean = self.y_mean.to(device)
        if hasattr(self, 'y_std') and self.y_std is not None:
            self.y_std = self.y_std.to(device)
        return self

import pickle
import numpy as np
from typing import Dict

class SimpleReplayPool:
    """簡單的經驗回放池，用於存儲過去的(s, a, s')"""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        max_size: int = 10000
    ):
        """
        初始化回放池
        
        Args:
            observation_dim: 觀察空間維度
            action_dim: 動作空間維度
            max_size: 池的最大容量
        """
        self.max_size = max_size
        self._size = 0
        self._pointer = 0
        
        # 初始化存儲空間
        self._observations = np.zeros((max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((max_size, action_dim), dtype=np.float32)
    
    def add_sample(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
    ) -> None:
        """
        添加一個樣本到回放池
        
        Args:
            observation: 當前觀察
            action: 執行的動作
            next_observation: 下一個觀察
        """
        self._observations[self._pointer] = observation
        self._actions[self._pointer] = action
        self._next_observations[self._pointer] = next_observation
        
        # 更新指針和大小
        self._pointer = (self._pointer + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)
    
    def random_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        隨機採樣一個批次的數據
        
        Args:
            batch_size: 批次大小
            
        Returns:
            包含批次數據的字典
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        
        return {
            'observations': self._observations[indices],
            'actions': self._actions[indices],
            'next_observations': self._next_observations[indices]
        }
    
    def __len__(self) -> int:
        """返回當前池中的樣本數量"""
        return self._size
    
    def clear(self) -> None:
        """清空回放池"""
        self._size = 0
        self._pointer = 0

    def save(self, path: str) -> None:
        """保存回放池"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> None:
        """加載回放池"""
        with open(path, 'rb') as f:
            return pickle.load(f)
