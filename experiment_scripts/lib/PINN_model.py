import torch
import torch.nn as nn
import numpy as np

class PINNModel_Tanh(nn.Module):
    """
    音響圧力 p(x, y, t) を近似するための多層パーセプトロン (MLP)。
    """
    def __init__(self, c=1.0):
        super(PINNModel_Tanh, self).__init__()
        self.c = c
        
        # ネットワーク層の定義 (3 -> 64 -> 64 -> 64 -> 1)
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class SirenLayer(nn.Module):
    """
    SIRENの単一層。nn.Linear と sin(omega_0 * x) 活性化関数、
    および専用の重み初期化をカプセル化します。
    """
    def __init__(self, in_features, out_features, bias=True, 
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        
        # 線形層
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # ★ SIREN専用の重み初期化を実行
        self.init_weights()

    def init_weights(self):
        """
        is_first フラグに基づいて、SIRENの初期化ルールを適用します。
        """
        with torch.no_grad():
            if self.is_first:
                # ★ 最初の層の初期化 (omega_0=30 の場合)
                # U[-1/n, 1/n]
                self.linear.weight.uniform_(-1.0 / self.in_features, 
                                            1.0 / self.in_features)
            else:
                # ★ 中間層の初期化 (omega_0=1 の場合)
                # c=6.0 は論文で推奨される定数
                c = 6.0 
                # U[-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0]
                self.linear.weight.uniform_(-np.sqrt(c / self.in_features) / self.omega_0, 
                                            np.sqrt(c / self.in_features) / self.omega_0)
        
            if self.linear.bias is not None:
                # バイアスはゼロで初期化
                self.linear.bias.data.fill_(0.0)

    def forward(self, input):
        # ★ sin(omega_0 * (Wx + b)) を計算
        return torch.sin(self.omega_0 * self.linear(input))
    
class PINNModel_SIREN(nn.Module):
    """
    音響圧力 p(x, y, t) を近似するためのSIRENモデル。
    """
    def __init__(self, c=1.0):
        super(PINNModel_SIREN, self).__init__()
        self.c = c
        
        # SIREN レイヤーでネットワークを構築
        self.net = nn.Sequential(
            # ★ 1. 最初の層: is_first=True, omega_0=30.0
            SirenLayer(in_features=3, out_features=64, is_first=True, omega_0=30.0),
            
            # ★ 2. 中間層: is_first=False, omega_0=1.0
            #    (論文では中間層に 6.0 を推奨する場合もありますが、
            #     会話の流れに沿って 1.0 を使用します)
            SirenLayer(in_features=64, out_features=64, is_first=False, omega_0=1.0),
            SirenLayer(in_features=64, out_features=64, is_first=False, omega_0=1.0),
            
            # ★ 3. 最終層: 通常の Linear (活性化関数なし)
            nn.Linear(64, 1)
        )
        
        # ★ 最終層の重みをSIRENの中間層スタイルで初期化
        self._initialize_final_layer()

    def _initialize_final_layer(self):
        """
        最終層は活性化関数がないため、個別に初期化します。
        """
        with torch.no_grad():
            final_layer = self.net[-1]
            in_features = final_layer.in_features
            c = 6.0
            omega_0 = 1.0 # 中間層と同じ設定
            
            # 中間層と同じ初期化を適用
            final_layer.weight.uniform_(-np.sqrt(c / in_features) / omega_0, 
                                        np.sqrt(c / in_features) / omega_0)
            if final_layer.bias is not None:
                final_layer.bias.data.fill_(0.0)

    def forward(self, x):
        return self.net(x)
    

class PINNModel_SIREN_altOmega(nn.Module):
    """
    音響圧力 p(x, y, t) を近似するためのSIRENモデル。
    """
    def __init__(self, c=1.0):
        super(PINNModel_SIREN_altOmega, self).__init__()
        self.c = c
        
        # SIREN レイヤーでネットワークを構築
        self.net = nn.Sequential(
            # ★ 1. 最初の層: is_first=True, omega_0=30.0
            SirenLayer(in_features=3, out_features=64, is_first=True, omega_0=10.0),
            
            # ★ 2. 中間層: is_first=False, omega_0=1.0
            #    (論文では中間層に 6.0 を推奨する場合もありますが、
            #     会話の流れに沿って 1.0 を使用します)
            SirenLayer(in_features=64, out_features=64, is_first=False, omega_0=1),
            SirenLayer(in_features=64, out_features=64, is_first=False, omega_0=1),
            
            # ★ 3. 最終層: 通常の Linear (活性化関数なし)
            nn.Linear(64, 1)
        )
        
        # ★ 最終層の重みをSIRENの中間層スタイルで初期化
        self._initialize_final_layer()

    def _initialize_final_layer(self):
        """
        最終層は活性化関数がないため、個別に初期化します。
        """
        with torch.no_grad():
            final_layer = self.net[-1]
            in_features = final_layer.in_features
            c = 6.0
            omega_0 = 1.0 # 中間層と同じ設定
            
            # 中間層と同じ初期化を適用
            final_layer.weight.uniform_(-np.sqrt(c / in_features) / omega_0, 
                                        np.sqrt(c / in_features) / omega_0)
            if final_layer.bias is not None:
                final_layer.bias.data.fill_(0.0)

    def forward(self, x):
        return self.net(x)