import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import time

# PyTorchのシード設定
torch.manual_seed(42)
np.random.seed(42)

# ================================================================
# 1. データセットと真の値の生成 (論文のセクション4に基づく)
# ================================================================

class AcousticTomographyDataset:
    """
    PINNトレーニング用のデータ（真の音場、射影測定値、コロケーションポイント）を生成するクラス。
    論文のセクション4の2D点音源の実験に基づいています。
    """
    def __init__(self, L=1.0, T=2.0, c=1.0, nl=20, nt=26, frequency=3.0, snr_db=20.0, pulse_width=0.1):
        # 物理パラメータ (無次元化を想定)
        self.L = L # 空間ドメインの半長 ([-L/2, L/2]を使用)
        self.T = T # 時間ドメインの最大値
        self.c = c # 音速 (無次元化された値)
        self.freq = frequency # 波の周波数 (パルス波の中心周波数として機能)

        # パルス波パラメータ
        self.pulse_width = pulse_width # ガウスパルスの幅を制御 (小さいほど鋭いパルス)

        # 測定/サンプリングパラメータ
        self.nl = nl # プローブビームの数 (論文と同じ20)
        self.nt = nt # 時間サンプルの数 (論文と同じ26)
        self.snr_db = snr_db # ノイズSNR (20 dB)
        
        # --- 修正: 音源位置 (論文の Figure 2 に基づく) ---
        self.src1 = torch.tensor([1.5 * L, 0.0])
        self.src2 = torch.tensor([-L, -L]) # 論文の設定 (L, L) に修正
        
        # --- 修正: 計算ドメインを音源とセンサー全体を覆うように拡張 ---
        # センサー: x=[-0.5L, 0.5L], y=[-0.5L, 0.5L]
        # 音源: (1.5L, 0), (L, L)
        # 必要な最小ドメイン: x=[-0.5L, 1.5L], y=[-0.5L, L]
        # マージンを持たせて設定
        self.domain_x = [-1.0 * L, 2.0 * L] # [-1.0, 2.0]
        self.domain_y = [-1.0 * L, 2.0 * L] # [-1.0, 2.0]
        # Lp計算時の音源からの最小除外距離
        self.source_exclusion_radius = 0.1 * L 

        # --- ファンビーム ジオメトリの再定義 ---
        
        # 発散点 (Source Point) P_s = (0, -L/2)
        self.source_x = 0.0
        self.source_y = -self.L / 2 
        
        # 検出ライン (Detector Line) P_d = (x, L/2)
        self.detector_y = self.L / 2
        
        # 検出ライン上の nl 個の均等に配置された検出点 (L/2 から -L/2 の範囲)
        self.detector_x_coords = torch.linspace(self.L / 2, -self.L / 2, self.nl).float()
        
        # L_v_params には検出器のX座標のみを格納する (ファンビームでは角度やシフトは不要)
        
        # 真の音場データの正規化に使用する最大振幅
        # P_maxを推定し、積分最大値 I_max を導出
        self.P_max = self._calculate_max_pressure()
        if self.P_max < 1e-4: 
             self.P_max = 1.0 # サンプリングが不安定な場合のフォールバック
        self.I_max = self.L * self.P_max * 1.5 # 積分の最大値の概算 (L * P_max * 安全係数)

    def _free_space_green_function_2d(self, R, t):
        """
        2D自由空間における点音源の応答。ガウス変調されたパルス波。
        """
        R = torch.clamp(R, min=1e-6)
        
        time_delay = R / self.c
        amplitude = 1.0 / torch.sqrt(R) 
        t_local = t - time_delay
        
        sigma = self.pulse_width / 2.0 
        gaussian_envelope = torch.exp(-(t_local**2) / (sigma**2))
        
        angular_freq = 2 * np.pi * self.freq
        carrier_wave = torch.sin(angular_freq * t_local)
        
        pressure = amplitude * gaussian_envelope * carrier_wave
        
        return pressure

    def true_pressure(self, X):
        """
        任意の時空間座標 (x, y, t) における真の音響圧力場を計算する。
        X: (N, 3) テンソル, X[:, 0]=x, X[:, 1]=y, X[:, 2]=t
        """
        r = X[:, :2]
        t = X[:, 2].unsqueeze(1)
        
        R1 = torch.linalg.norm(r - self.src1.to(r.device), dim=1).unsqueeze(1)
        P1 = self._free_space_green_function_2d(R1, t)
        
        R2 = torch.linalg.norm(r - self.src2.to(r.device), dim=1).unsqueeze(1)
        P2 = self._free_space_green_function_2d(R2, t)
        
        P_true = P1 + P2
        
        return P_true

    def _calculate_max_pressure(self, N=10000):
        """真の音場の最大振幅を推定する。"""
        # (コードの安定性を高めるため、今回は P_max を一時的に1.0に固定し、この関数は参照として残す)
        # --- 修正: 拡張されたドメインからサンプリング ---
        t = torch.rand(N, 1) * self.T
        x = (torch.rand(N, 1) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0])
        y = (torch.rand(N, 1) * (self.domain_y[1] - self.domain_y[0]) + self.domain_y[0])
        X_test = torch.cat([x, y, t], dim=1)

        with torch.no_grad():
            P_test = self.true_pressure(X_test)
        return P_test.abs().max().item()

    def generate_projection_data(self, beam_res=50):
        """
        射影データ V_v と、それに対応するパラメータを生成する。
        L_v_params の[:, 0] に検出器のX座標を格納する (ファンビーム)。
        """
        T_v = torch.linspace(0.0, self.T, self.nt).unsqueeze(1) 
        N_data = self.nl * self.nt
        s_points = torch.linspace(0.0, 1.0, beam_res).unsqueeze(1) # s=[0, 1]
        
        V_v_true = torch.zeros(N_data, 1) 
        L_v_params = torch.zeros(N_data, 2) # [det_x, 0]を格納
        
        idx = 0
        Ps = torch.tensor([self.source_x, self.source_y])
        
        for i in range(self.nl):
            det_x = self.detector_x_coords[i]
            
            L_v_params[idx:idx + self.nt, 0] = det_x
            
            # 発散点 Ps と検出点 Pd
            Pd = torch.tensor([det_x.item(), self.detector_y])
            
            # ベクトル V = Pd - Ps (ビームの方向)
            V = Pd - Ps
            L_beam = torch.linalg.norm(V).item()
            V_norm = V / L_beam
            
            # 積分パス s_points を L_beam にスケール
            s_path = s_points * L_beam 
            
            # 経路上の X, Y 座標を計算: P(s) = Ps + s * V_norm
            X_coords = Ps[0] + s_path * V_norm[0].item()
            Y_coords = Ps[1] + s_path * V_norm[1].item()
            
            ds_step = L_beam / (beam_res - 1)
            
            for j in range(self.nt):
                t = T_v[j]
                
                T_coords = torch.full_like(X_coords, t.item())
                P_input = torch.cat([X_coords, Y_coords, T_coords], dim=1)
                P_true_path = self.true_pressure(P_input)
                
                # 台形則による積分
                integral_val = torch.sum(P_true_path[1:-1]).item() 
                integral_val += (P_true_path[0].item() + P_true_path[-1].item()) / 2
                integral_val *= ds_step
                
                V_v_true[idx, 0] = integral_val
                idx += 1
                
        # ノイズの追加
        P_rms = torch.sqrt(torch.mean(V_v_true**2))
        noise_power = P_rms / (10**(self.snr_db / 20.0))
        noise = torch.randn_like(V_v_true) * noise_power.item()
        
        # V_v_trueとV_vを最大積分値 I_max で正規化する (L_vの安定化のため)
        V_v_true_scaled = V_v_true / self.I_max
        V_v_scaled = (V_v_true + noise) / self.I_max # ノイズ付き測定値も正規化
        
        T_v_all = T_v.repeat_interleave(self.nl, dim=0).view(-1, 1)
        
        return T_v_all, V_v_scaled, L_v_params # V_v_scaledを返す 

    def generate_pinn_collocation_points(self, N_p):
        """
        物理損失 L_p のためのランダムなコロケーションポイント X_p を生成する。
        --- 修正: 拡張ドメインからサンプリングし、音源近傍を除外する ---
        """
        device = self.src1.device # テンソルがCPU/GPUのどちらにあるか確認
        
        # ひとまず N_p 個の点を生成
        t = torch.rand(N_p, 1, device=device) * self.T
        x = (torch.rand(N_p, 1, device=device) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0])
        y = (torch.rand(N_p, 1, device=device) * (self.domain_y[1] - self.domain_y[0]) + self.domain_y[0])
        
        X_p = torch.cat([x, y, t], dim=1)
        xy = X_p[:, :2]

        # 音源からの距離を計算
        dist_sq1 = torch.sum((xy - self.src1.to(device))**2, dim=1)
        dist_sq2 = torch.sum((xy - self.src2.to(device))**2, dim=1)
        
        # 除外半径の2乗
        min_dist_sq = self.source_exclusion_radius**2
        
        # 除外マスク (どちらかの音源に近すぎる点)
        invalid_mask = (dist_sq1 < min_dist_sq) | (dist_sq2 < min_dist_sq)
        
        # 効率的なリサンプリング: invalid な点だけを再サンプリング
        while invalid_mask.any():
            n_invalid = invalid_mask.sum().item()
            
            # 新しい候補点を生成
            new_t = torch.rand(n_invalid, 1, device=device) * self.T
            new_x = (torch.rand(n_invalid, 1, device=device) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0])
            new_y = (torch.rand(n_invalid, 1, device=device) * (self.domain_y[1] - self.domain_y[0]) + self.domain_y[0])
            
            new_X_p = torch.cat([new_x, new_y, new_t], dim=1)
            new_xy = new_X_p[:, :2]

            # invalid な点を新しい候補点で置き換え
            X_p[invalid_mask] = new_X_p

            # 新しい点の距離を再計算
            new_dist_sq1 = torch.sum((new_xy - self.src1.to(device))**2, dim=1)
            new_dist_sq2 = torch.sum((new_xy - self.src2.to(device))**2, dim=1)
            
            # invalid_mask を更新 (置き換えた点のみをチェック)
            invalid_mask[invalid_mask.clone()] = (new_dist_sq1 < min_dist_sq) | (new_dist_sq2 < min_dist_sq)

        # X_p[:, :2] = ... (これは間違い。X_p全体を置き換える必要がある)
        # X_p[:, 2] = ...
        
        # 時間 t 座標を [0, T] にスケーリング
        # X_p[:, 2] = X_p[:, 2] * self.T (これはリサンプリングループの中で処理済み)
        
        return X_p

    # --- 削除: 初期条件は L_v と矛盾するため使用しない ---
    # def generate_initial_collocation_points(self, N_ic): ...


    def generate_validation_data(self, sl_x, sl_y, n_frames, frames=None):
        """
        検証用の時空間グリッドと真の圧力場を生成する。
        --- 修正: 論文の図2(c,d)の点線領域 ([-L/2, L/2]) で評価する ---
        """
        x = torch.linspace(-self.L/2, self.L/2, sl_x)
        y = torch.linspace(-self.L/2, self.L/2, sl_y)
        
        # フレーム時間の決定
        if frames is None:
             frames = np.linspace(0.0, self.T, n_frames, endpoint=False)
        else:
             n_frames = len(frames)
        
        # グリッドの生成
        XX, YY = torch.meshgrid(x, y, indexing='xy')
        
        # 全てのフレームを結合
        X_val_list = []
        P_val_true_list = []
        
        for t_frame in frames:
            TT = torch.full_like(XX, t_frame)
            # (sl_x * sl_y, 3) の形状
            X_frame = torch.stack([XX.flatten(), YY.flatten(), TT.flatten()], dim=1)
            X_val_list.append(X_frame)
            
            with torch.no_grad():
                P_val_true_list.append(self.true_pressure(X_frame))
                
        X_val = torch.cat(X_val_list, dim=0) # (n_frames * sl_x * sl_y, 3)
        P_val_true = torch.cat(P_val_true_list, dim=0) # (n_frames * sl_x * sl_y, 1)

        return X_val, P_val_true, frames

# ================================================================
# 2. PINN モデルと損失関数
# ================================================================

class PINNModel(nn.Module):
    """
    音響圧力 p(x, y, t) を近似するための多層パーセプトロン (MLP)。
    --- 修正: 入力正規化機能を追加 ---
    """
    def __init__(self, c=1.0, domain_min=None, domain_max=None):
        super(PINNModel, self).__init__()
        self.c = c
        
        # --- 追加: 入力正規化のためのドメイン範囲 ---
        if domain_min is None or domain_max is None:
             # デフォルト値 (安全のため)
             domain_min = torch.tensor([0.0, -1.0, -1.0])
             domain_max = torch.tensor([2.0, 2.0, 2.0])
             
        # 登録バッファ (状態の一部として保存されるが、パラメータではない)
        # (1, 3) の形状で登録
        self.register_buffer("domain_min", domain_min.float().unsqueeze(0)) 
        self.register_buffer("domain_max", domain_max.float().unsqueeze(0))
        
        # 範囲 (max - min)
        domain_range = self.domain_max - self.domain_min
        # ゼロ除算を避ける
        domain_range[domain_range == 0] = 1.0 
        self.register_buffer("domain_range", domain_range)

        # ネットワーク (変更なし)
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
        # x の形状は (N, 3)
        
        # --- 追加: 入力正規化 (Min-Max スケーリング) ---
        
        # (x - min) / (max - min) -> [0, 1] の範囲にスケーリング
        x_scaled_01 = (x - self.domain_min) / self.domain_range
        
        # 2.0 * x - 1.0 -> [-1, 1] の範囲にスケーリング (Tanh に最適)
        x_scaled = 2.0 * x_scaled_01 - 1.0
        
        # 正規化された入力でネットワークを通過
        return self.net(x_scaled)

def calculate_projection_integral_from_nn(model, L_v_params, T_v, dataset, beam_res=50):
    """
    ネットワーク出力 p_NN をビーム経路に沿って数値積分し、射影 h_j[f_NN] を計算する。
    修正: ファンビーム ジオメトリに対応。
    """
    N_data = T_v.shape[0]
    
    # --- 積分パスの設定 ---
    
    # 積分パラメータ s を [0, 1] で定義
    s_points = torch.linspace(0.0, 1.0, beam_res).to(T_v.device).float().unsqueeze(1) # (beam_res, 1)
    
    # Detector X 座標 (L_v_params[:, 0]) と時間を beam_res 倍にリピート
    T_v_rep = T_v.repeat(1, beam_res).view(-1, 1) 
    det_x_rep = L_v_params[:, 0].unsqueeze(1).repeat(1, beam_res).view(-1, 1)
    
    # s_points: (beam_res, 1) -> (N_data, beam_res) -> (N_data * beam_res, 1)
    s_points_rep = s_points.repeat(N_data, 1) 
    
    # --- ファンビーム座標計算 (P(s) = Ps + s * V_norm * L_beam) ---
    
    Ps_x = dataset.source_x
    Ps_y = dataset.source_y
    Pd_y = dataset.detector_y
    Pd_x_rep = det_x_rep
    
    # ベクトル V = Pd - Ps
    Vx_rep = Pd_x_rep - Ps_x
    Vy_rep = Pd_y - Ps_y
    
    # L_beam の計算 (各ビームごとに異なる長さ)
    L_beam_rep = torch.sqrt(Vx_rep**2 + Vy_rep**2)
    
    # 積分パス上のパラメータ s を L_beam_rep にスケール
    s_path = s_points_rep * L_beam_rep 
    
    # 単位ベクトル V_norm = V / L_beam
    L_beam_norm_safe = torch.clamp(L_beam_rep, min=1e-8)
    V_norm_x_rep = Vx_rep / L_beam_norm_safe
    V_norm_y_rep = Vy_rep / L_beam_norm_safe
    
    # 経路上の X, Y 座標を計算
    X_coords = Ps_x + s_path * V_norm_x_rep
    Y_coords = Ps_y + s_path * V_norm_y_rep
    
    # ネットワーク入力 (N_data * beam_res, 3)
    P_input = torch.cat([X_coords, Y_coords, T_v_rep], dim=1)
    
    # ネットワークによる圧力推定 (微分グラフを維持)
    P_nn_path = model(P_input) # (N_data * beam_res, 1)
    
    # P_nn_pathを (N_data, beam_res) に整形し直す
    P_nn_path_reshaped = P_nn_path.view(N_data, beam_res)
    
    # --- 台形則による積分（バッチ処理） ---
    
    # 積分ステップ ds_step は各ビームごとに異なる (L_beam / (beam_res - 1))
    ds_step_rep = L_beam_rep / (beam_res - 1)
    
    # ds_step を N_data のテンソルに整形
    ds_step_data = ds_step_rep.view(N_data, beam_res)[:, 0].unsqueeze(1)
    
    # 台形則の計算
    integral_sum = torch.sum(P_nn_path_reshaped[:, 1:-1], dim=1).unsqueeze(1)
    integral_val = integral_sum + (P_nn_path_reshaped[:, 0].unsqueeze(1) + P_nn_path_reshaped[:, -1].unsqueeze(1)) / 2
    integral_val *= ds_step_data

    # L_vのターゲット V_v が I_max で正規化されているため、integral_val も正規化する
    integral_val_scaled = integral_val / dataset.I_max

    return integral_val_scaled # (N_data, 1)

# --- 修正: L_ic に関連する引数 (X_ic, lambda_ic) を削除 ---
def pinn_loss_function(model, X_p, T_v, V_v, L_v_params, dataset, lambda_param=1e4, lp_weight=1.0):
    """
    PINNの総合損失関数 L(theta) = L_p + lambda * L_v を計算する。
    """
    # --- 物理損失 L_p の計算 (波動方程式の残差) ---
    # X_p は音源近傍が除外された点が渡される
    X_p.requires_grad_(True) 
    p_nn = model(X_p)
    
    # 1階導関数
    gradients = grad(p_nn, X_p, grad_outputs=torch.ones_like(p_nn), create_graph=True)[0]
    dp_dx = gradients[:, 0].unsqueeze(1)
    dp_dy = gradients[:, 1].unsqueeze(1)
    dp_dt = gradients[:, 2].unsqueeze(1)

    # 2階導関数
    d2p_dx2 = grad(dp_dx, X_p, grad_outputs=torch.ones_like(dp_dx), create_graph=True)[0][:, 0].unsqueeze(1)
    d2p_dy2 = grad(dp_dy, X_p, grad_outputs=torch.ones_like(dp_dy), create_graph=True)[0][:, 1].unsqueeze(1)
    d2p_dt2 = grad(dp_dt, X_p, grad_outputs=torch.ones_like(dp_dt), create_graph=True)[0][:, 2].unsqueeze(1)

    # 波動方程式の残差 r_p (式(1)に相当)
    # 音源 q=0 の領域での残差を計算
    c_squared = model.c**2
    residual_p = (d2p_dx2 + d2p_dy2) - (1.0 / c_squared) * d2p_dt2
    
    # 物理損失 L_p (式(7))
    L_p = torch.mean(residual_p**2)

    # --- 削除: 初期条件損失 L_ic の計算 ---
    # L_ic = torch.tensor(0.0).to(X_p.device) ...
    
    # --- データ損失 L_v の計算 (射影誤差) ---
    
    # ネットワーク出力からの射影積分の計算 h_j[f_NN]
    H_j_p_nn = calculate_projection_integral_from_nn(model, L_v_params, T_v, dataset)
    
    # データ損失 L_v (式(8))
    L_v = torch.mean((V_v - H_j_p_nn)**2)
    
    # --- 修正: 総合損失の計算 (L_p に重みを適用) ---
    total_loss = (lp_weight * L_p) + (lambda_param * L_v)
    
    return total_loss, L_p, L_v

# --- 削除: Sine() クラス (nn.Tanh() を使用するため不要) ---
# class Sine(nn.Module): ...

# ================================================================
# 3. 評価と可視化 (論文の図2, 3を再現)
# ================================================================

def evaluate_and_plot(model, dataset, T_v_full, V_v_full, sl=100):
    """
    最終的な結果を評価し、論文の図2（波面スナップショット）と図3（定量的誤差）を模倣した
    プロットを生成する。
    また、測定された射影データ（V_v）を可視化する（図4）。
    """
    device = next(model.parameters()).device
    model.eval()
    
    # --- 図4: 射影データ（シノグラム）の可視化 ---
    
    # データ V_v は (nl*nt, 1) の形状で、ビームインデックスと時間が交互になっている
    # 整形: (nt, nl) に変換。時間軸が縦、ビームインデックスが横になるように。
    V_v_reshaped = V_v_full.view(dataset.nt, dataset.nl).cpu().numpy()
    
    # 時間軸のラベル設定
    time_labels = np.linspace(0.0, dataset.T, dataset.nt)
    
    # ビームインデックスは 0 から nl-1
    beam_indices = np.arange(dataset.nl)
    
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    
    # extent: [xmin, xmax, ymin, ymax]。原点 (0, 0) は左上になるように設定
    im = ax4.imshow(
        V_v_reshaped,
        aspect='auto',
        cmap='seismic',
        interpolation='none',
        extent=[beam_indices.min(), beam_indices.max(), time_labels.max(), time_labels.min()]
    )
    
    ax4.set_title('Measured Projection Data (Synogram) - V_v')
    ax4.set_xlabel('Beam Index (Detector X Coordinate Index)')
    ax4.set_ylabel('Time (t)')
    plt.colorbar(im, ax=ax4, label='Line Integral Value')
    plt.show()

    
    # --- 定量的誤差（NMSE, CS）の時系列計算 ---
    time_points_val = np.linspace(0.0, dataset.T, 20) 
    nmse_series = []
    cs_series = []
    
    with torch.no_grad():
        for t_val in time_points_val:
            # t_valでの単一のスナップショットを生成 (評価領域 [-L/2, L/2])
            X_time, P_true_time, _ = dataset.generate_validation_data(sl, sl, 1, frames=[t_val])
            
            # 予測
            P_est_time = model(X_time.to(device)).cpu()
            P_true_time = P_true_time.cpu()
            
            # メトリック計算 (NMSEとCSのユーティリティ関数をインラインで定義)
            def calculate_nmse(p_true, p_est):
                # 論文に従い、最大値と最小値の差で正規化
                p_range = p_true.max() - p_true.min()
                if p_range.item() < 1e-6: 
                     # t=0.0 付近で P_true がほぼゼロで p_range が小さい場合、NMSEが発散するのを防ぐ
                     # この場合、純粋なRMSEを返すか、固定の最小レンジを使用
                     return torch.sqrt(torch.mean((p_true - p_est)**2)).item() / dataset.P_max 
                rmse = torch.sqrt(torch.mean((p_true - p_est)**2))
                return (rmse / p_range).item()

            def calculate_cs(p_true, p_est):
                numerator = torch.sum(p_true * p_est)
                denominator = torch.sqrt(torch.sum(p_true**2) * torch.sum(p_est**2))
                if denominator.item() < 1e-9: return 1.0
                return (numerator / denominator).item()

            nmse_series.append(calculate_nmse(P_true_time, P_est_time))
            cs_series.append(calculate_cs(P_true_time, P_est_time))

    # --- 図3: NMSE/CSの時系列プロット ---
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 5))
    
    ax3[0].plot(time_points_val, nmse_series, label='PINN NMSE', marker='o')
    ax3[0].set_title('Normalized Mean Square Error (NMSE) over Time')
    ax3[0].set_xlabel('Time (t)')
    ax3[0].set_ylabel('NMSE')
    ax3[0].set_xlim(1, 2)
    ax3[0].grid(True)
    
    ax3[1].plot(time_points_val, cs_series, label='PINN CS', marker='o', color='red')
    ax3[1].set_title('Cosine Similarity (CS) over Time')
    ax3[1].set_xlabel('Time (t)')
    ax3[1].set_ylabel('CS')
    ax3[1].set_xlim(1, 2)
    ax3[1].set_ylim(-1, 1.05)
    ax3[1].grid(True)
    
    fig3.suptitle("Quantitative Error Metrics (Fig. 3 Emulation)", fontsize=14)
    plt.show()

    # --- 図2: 2D波面スナップショットのプロット ---
    
    # 論文の図2に合わせたフレーム（時間の開始、中間、終了付近など）
    # トレーニングが進んでいれば、これらのフレームで波面が確認できます
    frames = np.linspace(dataset.T * 0.5, dataset.T * 0.9, 4).tolist()
    
    X_val, P_val_true, frames = dataset.generate_validation_data(sl, sl, len(frames), frames=frames)
    
    # 推定値を計算し、致命的なエラーを回避するために.detach()を呼び出す
    with torch.no_grad():
        P_val_est = model(X_val.to(device)).cpu()
    
    P_val_true_reshaped = P_val_true.view(len(frames), sl, sl).numpy()
    
    # 修正: .detach().numpy() を使用して勾配を切断
    P_val_est_reshaped = P_val_est.detach().view(len(frames), sl, sl).numpy()
    
    # 全スナップショットの共通カラーマップ範囲を決定 (視覚比較に重要)
    v_min = min(P_val_true_reshaped.min(), P_val_est_reshaped.min())
    v_max = max(P_val_true_reshaped.max(), P_val_est_reshaped.max())
    
    fig2, axes = plt.subplots(len(frames), 2, figsize=(8, 4 * len(frames)))
    
    for i in range(len(frames)):
        
        # 1列目: 参照 (Reference)
        ax_true = axes[i, 0]
        im_true = ax_true.imshow(
            P_val_true_reshaped[i], 
            cmap='seismic', 
            vmin=v_min, 
            vmax=v_max, 
            extent=[-dataset.L/2, dataset.L/2, -dataset.L/2, dataset.L/2]
        )
        ax_true.set_title(f"Reference (t={frames[i]:.2f})")
        ax_true.set_aspect('equal')
        ax_true.set_xlabel('X')
        ax_true.set_ylabel('Y')
        
        # 2列目: PINN推定 (Proposal/Estimate)
        ax_est = axes[i, 1]
        im_est = ax_est.imshow(
            P_val_est_reshaped[i], 
            cmap='seismic', 
            vmin=v_min, 
            vmax=v_max, 
            extent=[-dataset.L/2, dataset.L/2, -dataset.L/2, dataset.L/2]
        )
        ax_est.set_title(f"PINN Estimate (t={frames[i]:.2f})")
        ax_est.set_aspect('equal')
        ax_est.set_xlabel('X')
        
        # カラーバーの追加 (右側のプロットにのみ)
        plt.colorbar(im_est, ax=ax_est, fraction=0.046, pad=0.04)

    fig2.suptitle("Pressure Field Snapshots (Fig. 2 Emulation)", fontsize=14)
    plt.tight_layout()
    plt.show()

# ================================================================
# 4. メイン実行ブロック
# ================================================================

if __name__ == '__main__':
    # --- ハイパーパラメータ設定 (論文の記述に基づく) ---
    N_EPOCHS = 10000 # 論文は 5 * 10^4 エポック
    N_COLLOCATION = 1000 # L_p ポイント (論文は100だが、ドメインが広がったため増やすことを推奨)
    # N_IC = 1000 # --- 削除 ---
    LR = 1e-4
    LAMBDA = 1e6 # --- 修正: 論文の推奨値 (1e6 -> 1e4) ---
    # LAMBDA_IC = 1e5 # --- 削除 ---

    # --- 追加: L_p の重みスケジューリング設定 ---
    LP_WEIGHT_START = 0.01  # 学習開始時の L_p の重み
    LP_WEIGHT_END = 1.0     # 最終的な L_p の重み
    # L_p の重みを 1.0 に戻すまでのエポック数 (例: 全体の40%)
    LP_SCHEDULE_EPOCHS = int(N_EPOCHS * 0.4) # 20000エポック
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # --- データセットとモデルの準備 ---
    
    # 論文の実験パラメータ
    dataset = AcousticTomographyDataset(L=1.0, T=2.0, c=1.0, nl=20, nt=26, frequency=3.0, snr_db=20.0, pulse_width=0.3)
    
    # L_v のターゲットデータ (時間と測定値) はトレーニング開始時に一度生成
    print("射影データ生成中...")
    T_v, V_v, L_v_params = dataset.generate_projection_data(beam_res=50)
    T_v_full = T_v.to(device) # フルデータセット (時間)
    V_v_full = V_v.to(device) # フルデータセット (測定値)
    L_v_params_full = L_v_params.to(device) # フルデータセット (パラメータ)
    print(f"射影データポイント数: {V_v_full.shape[0]}")
    
    # --- 修正: モデル初期化時にドメイン情報を渡す ---
    
    # 正規化に使用するドメイン情報をテンソルとして準備
    domain_min_tensor = torch.tensor([
        0.0, # t_min
        dataset.domain_x[0], # x_min
        dataset.domain_y[0]  # y_min
    ]).to(device)
    
    domain_max_tensor = torch.tensor([
        dataset.T, # t_max
        dataset.domain_x[1], # x_max
        dataset.domain_y[1]  # y_max
    ]).to(device)

    # モデルとオプティマイザの初期化
    model = PINNModel(
        c=dataset.c, 
        domain_min=domain_min_tensor, 
        domain_max=domain_max_tensor
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # --- トレーニングループ ---
    # --- 修正: print 文から L_ic を削除 ---
    print(f"\n--- PINN トレーニング開始 (λ={LAMBDA:.1e}, Epochs={N_EPOCHS}) ---")
    
    start_time = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # L_p のための新しいコロケーションポイントを各エポックでサンプリング
        # (音源近傍は自動的に除外される)
        X_p = dataset.generate_pinn_collocation_points(N_COLLOCATION).to(device)

        # --- 追加: 現在のエポックに基づいて L_p の重みを計算 ---
        if epoch < LP_SCHEDULE_EPOCHS:
            # 線形に重みを増加
            lp_weight = LP_WEIGHT_START + (LP_WEIGHT_END - LP_WEIGHT_START) * (epoch / LP_SCHEDULE_EPOCHS)
        else:
            lp_weight = LP_WEIGHT_END
        
        # --- 修正: 損失の計算 (lp_weight を渡す) ---
        total_loss, L_p, L_v = pinn_loss_function(
            model, 
            X_p, 
            T_v_full, 
            V_v_full, 
            L_v_params_full, 
            dataset, 
            lambda_param=LAMBDA,
            lp_weight=lp_weight # 計算した L_p の重みを渡す
        )
        
        # バックプロパゲーションと最適化
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            # --- 修正: print 文から L_ic を削除 ---
            print(f"Epoch {epoch}/{N_EPOCHS} | Total Loss: {total_loss.item():.6e} | "
                  f"L_p: {L_p.item():.6e} | L_v: {L_v.item():.6e} | Time: {elapsed:.2f}s")
            start_time = time.time() # 1000エポックごとの時間計測をリセット
    
    print("\n--- トレーニング完了 ---")
    
    # --- 最終評価とプロット ---
    # 波面スナップショットと定量的誤差を表示します。
    # 修正: V_v_fullをevaluate_and_plotに渡す
    evaluate_and_plot(model, dataset, T_v_full, V_v_full, sl=100)