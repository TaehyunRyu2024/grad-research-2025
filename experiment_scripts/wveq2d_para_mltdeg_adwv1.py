import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import argparse
import glob

# ★ 警告を非表示にする設定
import warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")

# PyTorchのシード設定
torch.manual_seed(42)
np.random.seed(42)

# ================================================================
# 1. データセットと真の値の生成 (変更なし)
# ================================================================

class AcousticTomographyDataset:
    """
    PINNトレーニング用のデータ生成クラス。
    ★ 修正: 平行ビーム・回転スキャン (Parallel Beam Tomography) に変更
    """
    def __init__(self, L=1.0, T=2.0, c=1.0, nl=20, n_angles=18, nt=26, frequency=3.0, snr_db=20.0):
        # 物理パラメータ
        self.L = L 
        self.T = T 
        self.c = c 
        self.freq = frequency 

        # 測定パラメータ
        self.n_rays = nl       # 1角度あたりの平行ビーム数 (旧 nl)
        self.n_angles = n_angles # ★ 追加: 角度の分割数 (0〜180度)
        self.nt = nt           # 時間サンプル数
        self.snr_db = snr_db 
        
        # 音源位置 (シミュレーション用の真の音源 - 変更なし)
        self.src1 = torch.tensor([1.5 * L, 0.0])
        self.src2 = torch.tensor([-L, L])
        self.src3 = torch.tensor([-0.4, -0.4])
        
        # --- ★ 修正: 回転平行ビームのジオメトリ設定 ---
        
        # 角度範囲 [0, pi)
        angles_np = np.linspace(0, np.pi, self.n_angles, endpoint=False)
        self.angles = torch.tensor(angles_np, dtype=torch.float32)
        
        # センサーアレイの幅
        # 正方形領域 [-L/2, L/2] を回転してもカバーできるよう、対角線長より大きく取る
        scan_width = self.L * 1.5 
        
        # 中心からのオフセット u (各角度でのレイの位置)
        self.offsets = torch.linspace(-scan_width/2, scan_width/2, self.n_rays)
        
        # ソースと検出器の「回転中心からの距離」
        # 領域外に配置するため、十分に大きく取る
        self.radius = self.L * 3 / 4 

        # 真の音場データの正規化に使用する最大振幅
        self.P_max = self._calculate_max_pressure()
        if self.P_max < 1e-4: self.P_max = 1.0 
        self.I_max = self.L * self.P_max * 1.5 

    def _free_space_green_function_2d(self, R, t):
        # (変更なし)
        R = torch.clamp(R, min=1e-6)
        time_delay = R / self.c
        amplitude = 1.0 / torch.sqrt(R) 
        t_local = t - time_delay
        pf = np.pi * self.freq
        pf2 = pf**2
        ricker_term = (1 - 2 * pf2 * t_local**2) * torch.exp(-pf2 * t_local**2)
        return amplitude * ricker_term/5

    def true_pressure(self, X):
        # (変更なし)
        r = X[:, :2]
        t = X[:, 2].unsqueeze(1)
        R1 = torch.linalg.norm(r - self.src1.to(r.device), dim=1).unsqueeze(1)
        P1 = self._free_space_green_function_2d(R1, t)
        R2 = torch.linalg.norm(r - self.src2.to(r.device), dim=1).unsqueeze(1)
        P2 = self._free_space_green_function_2d(R2, t)
        R3 = torch.linalg.norm(r - self.src3.to(r.device), dim=1).unsqueeze(1)
        P3 = self._free_space_green_function_2d(R3, t)

        return P1 + P2 + P3

    def _calculate_max_pressure(self, N=10000):
        # (変更なし)
        X_test = torch.rand(N, 3) * self.L - self.L/2
        X_test[:, 2] = X_test[:, 2] * (self.T / self.L) + self.T/2
        with torch.no_grad():
            P_test = self.true_pressure(X_test)
        max_p = P_test.abs().max().item()
        return max(max_p, 1.0)

    def generate_projection_data(self, beam_res=50):
        """
        射影データ V_v を生成する。
        ★ 修正: 角度(angle) と オフセット(u) の2重ループで並列ビームを生成
        """
        T_v = torch.linspace(0.0, self.T, self.nt).unsqueeze(1) 
        
        # 総データ数 = 角度数 * レイ数 * 時間数
        N_spatial = self.n_angles * self.n_rays
        N_data = N_spatial * self.nt
        
        s_points = torch.linspace(0.0, 1.0, beam_res).unsqueeze(1)
        
        V_v_true = torch.zeros(N_data, 1) 
        
        # パラメータ保存用: [theta, u]
        L_v_params = torch.zeros(N_data, 2)
        
        idx = 0
        
        # 外側ループ: 角度 theta
        for theta in self.angles:
            # 方向ベクトル
            # レイの進行方向 vec_d = (cos theta, sin theta)
            ray_dx = torch.cos(theta)
            ray_dy = torch.sin(theta)
            
            # センサー列の方向（進行方向に垂直） vec_u = (-sin theta, cos theta)
            perp_dx = -torch.sin(theta)
            perp_dy = torch.cos(theta)
            
            # 内側ループ: オフセット u (平行ビーム)
            for u in self.offsets:
                
                # パラメータの記録
                L_v_params[idx:idx+self.nt, 0] = theta
                L_v_params[idx:idx+self.nt, 1] = u
                
                # ビームの始点 (Ps) と終点 (Pd) の計算
                # 中心(0,0)から u 離れた軸上の点に対し、radius 分だけ戻る/進む
                # Ps = (u * vec_u) - (radius * vec_d)
                # Pd = (u * vec_u) + (radius * vec_d)
                
                center_x = u * perp_dx
                center_y = u * perp_dy
                
                Ps_x = center_x - self.radius * ray_dx
                Ps_y = center_y - self.radius * ray_dy
                
                Pd_x = center_x + self.radius * ray_dx
                Pd_y = center_y + self.radius * ray_dy
                
                # --- 以下、積分の実行 (座標変換以外は共通ロジック) ---
                
                # ベクトル V = Pd - Ps (長さは 2*radius で固定だが、汎用性のため計算)
                Vx = Pd_x - Ps_x
                Vy = Pd_y - Ps_y
                L_beam = torch.sqrt(Vx**2 + Vy**2).item()
                
                # 積分パス
                X_coords = Ps_x + s_points * Vx * L_beam # s_points scaled by L_beam logic? 
                # 元のコードのロジックに合わせる: s_points * L_beam * V_norm
                # V_norm = V / L_beam なので、 s_points * V でOK
                X_coords = Ps_x + s_points * Vx
                Y_coords = Ps_y + s_points * Vy
                
                ds_step = L_beam / (beam_res - 1)
                
                for j in range(self.nt):
                    t = T_v[j]
                    T_coords = torch.full_like(X_coords, t.item())
                    P_input = torch.cat([X_coords, Y_coords, T_coords], dim=1)
                    P_true_path = self.true_pressure(P_input)
                    
                    # 台形積分
                    integral_val = torch.sum(P_true_path[1:-1]).item() 
                    integral_val += (P_true_path[0].item() + P_true_path[-1].item()) / 2
                    integral_val *= ds_step
                    
                    V_v_true[idx, 0] = integral_val
                    idx += 1
        
        # 正規化とノイズ付加
        actual_I_max = torch.max(torch.abs(V_v_true))
        if actual_I_max.item() < 1e-6:
            self.I_max = 1.0
        else:
            self.I_max = actual_I_max.item() 
        print(f"I_max: {self.I_max:.4e}")

        P_rms = torch.sqrt(torch.mean(V_v_true**2))
        noise_power = P_rms / (10**(self.snr_db / 20.0))
        noise = torch.randn_like(V_v_true) * noise_power.item()
        
        V_v_scaled = (V_v_true + noise) / self.I_max 
        T_v_all = T_v.repeat(N_spatial, 1) 
        
        return T_v_all, V_v_scaled, L_v_params

    # 以下は変更なし
    def generate_pinn_collocation_points(self, N_p):
        X_p = torch.rand(N_p, 3)
        X_p[:, :2] = X_p[:, :2] * self.L - self.L/2
        X_p[:, 2] = X_p[:, 2] * self.T
        return X_p

    def generate_initial_collocation_points(self, N_ic):
        X_ic = torch.rand(N_ic, 3)
        X_ic[:, :2] = X_ic[:, :2] * self.L - self.L/2
        X_ic[:, 2] = 0.0
        return X_ic

    def generate_validation_data(self, sl_x, sl_y, n_frames, frames=None):
        x = torch.linspace(-self.L/2, self.L/2, sl_x)
        y = torch.linspace(-self.L/2, self.L/2, sl_y)
        if frames is None:
             frames = np.linspace(0.0, self.T, n_frames, endpoint=False)
        else:
             n_frames = len(frames)
        XX, YY = torch.meshgrid(x, y, indexing='xy')
        X_val_list = []
        P_val_true_list = []
        for t_frame in frames:
            TT = torch.full_like(XX, t_frame)
            X_frame = torch.stack([XX.flatten(), YY.flatten(), TT.flatten()], dim=1)
            X_val_list.append(X_frame)
            with torch.no_grad():
                P_val_true_list.append(self.true_pressure(X_frame))
        X_val = torch.cat(X_val_list, dim=0) 
        P_val_true = torch.cat(P_val_true_list, dim=0) 
        return X_val, P_val_true, frames
    
    def plot_geometry(self, save_path=None):
        """
        現在の平行ビームの設定を可視化して保存するメソッド
        """
        print("ジオメトリ構成を可視化中...")
        
        # 混雑を防ぐため最大4つの角度だけピックアップして表示
        if self.n_angles <= 4:
            indices = range(self.n_angles)
        else:
            indices = np.linspace(0, self.n_angles - 1, 4, dtype=int)
            
        n_plots = len(indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), constrained_layout=True)
        if n_plots == 1: axes = [axes]
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            theta = self.angles[idx]
            theta_deg = np.degrees(theta.item())
            
            # ROI (計算領域)
            rect = plt.Rectangle((-self.L/2, -self.L/2), self.L, self.L, 
                               linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            
            # 計算用パラメータ
            ray_dx = torch.cos(theta).item()
            ray_dy = torch.sin(theta).item()
            perp_dx = -torch.sin(theta).item()
            perp_dy = torch.cos(theta).item()
            
            # ビームの描画
            for u_idx, u in enumerate(self.offsets):
                u_val = u.item()
                center_x = u_val * perp_dx
                center_y = u_val * perp_dy
                
                Ps_x = center_x - self.radius * ray_dx
                Ps_y = center_y - self.radius * ray_dy
                Pd_x = center_x + self.radius * ray_dx
                Pd_y = center_y + self.radius * ray_dy
                
                ax.plot([Ps_x, Pd_x], [Ps_y, Pd_y], 'g-', alpha=0.3, linewidth=1)
                
                # 端のレイだけ点を描画
                if u_idx == 0 or u_idx == len(self.offsets)-1:
                    ax.plot(Ps_x, Ps_y, 'bo', markersize=5) 
                    ax.plot(Pd_x, Pd_y, 'ro', markersize=5) 

            # 矢印
            ax.arrow(0, 0, ray_dx*0.5, ray_dy*0.5, head_width=0.1, head_length=0.1, fc='k', ec='k')
            
            ax.set_title(f"Angle Index {idx}: {theta_deg:.1f}°")
            ax.set_xlim(-self.radius*1.2, self.radius*1.2)
            ax.set_ylim(-self.radius*1.2, self.radius*1.2)
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
        
        fig.suptitle(f"Parallel Beam Geometry: {self.n_angles} Angles", fontsize=14)
        
        if save_path:
            plt.savefig(save_path)
            print(f"★ ジオメトリ確認画像を保存しました: {save_path}")
            plt.close(fig)
        else:
            plt.show()

# ================================================================
# 2. PINN モデルと損失関数 (変更なし)
# ================================================================

class PINNModel(nn.Module):
    """
    音響圧力 p(x, y, t) を近似するための多層パーセプトロン (MLP)。
    """
    def __init__(self, c=1.0):
        super(PINNModel, self).__init__()
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

def calculate_projection_integral_from_nn(model, L_v_params, T_v, dataset, beam_res=50):
    """
    ニューラルネットワーク出力からの積分計算。
    平行ビームジオメトリ (Theta, u) に対応
    """
    N_data = T_v.shape[0]
    
    # L_v_params = [theta, u]
    theta = L_v_params[:, 0].unsqueeze(1) # (N_data, 1)
    u     = L_v_params[:, 1].unsqueeze(1) # (N_data, 1)
    
    # --- ジオメトリベクトルの計算 (バッチ処理) ---
    ray_dx = torch.cos(theta)
    ray_dy = torch.sin(theta)
    perp_dx = -torch.sin(theta)
    perp_dy = torch.cos(theta)
    
    # 始点 Ps と 終点 Pd の計算
    radius = dataset.radius
    center_x = u * perp_dx
    center_y = u * perp_dy
    
    Ps_x = center_x - radius * ray_dx
    Ps_y = center_y - radius * ray_dy
    Pd_x = center_x + radius * ray_dx
    Pd_y = center_y + radius * ray_dy
    
    # ベクトル V = Pd - Ps
    Vx = Pd_x - Ps_x
    Vy = Pd_y - Ps_y
    L_beam = torch.sqrt(Vx**2 + Vy**2) 
    
    # --- 積分パスの設定 (★ここを修正) ---
    # s_points を (1, beam_res) の形状で作成
    s_points = torch.linspace(0.0, 1.0, beam_res).to(T_v.device).float().unsqueeze(0) 
    
    # 各パラメータを (N_data, beam_res) の形状に拡張
    Ps_x_rep = Ps_x.repeat(1, beam_res)
    Ps_y_rep = Ps_y.repeat(1, beam_res)
    Vx_rep   = Vx.repeat(1, beam_res)
    Vy_rep   = Vy.repeat(1, beam_res)
    
    # 時間Tも拡張してからフラット化する準備 (N_data, beam_res)
    T_v_rep_2d = T_v.repeat(1, beam_res)
    
    # s_points をデータ数分だけ縦に複製 -> (N_data, beam_res)
    s_points_rep = s_points.repeat(N_data, 1)
    
    # 経路座標: P(s) = Ps + s * V
    # 計算時は全て (N_data, beam_res) の形状で行う
    X_coords = (Ps_x_rep + s_points_rep * Vx_rep).view(-1, 1)
    Y_coords = (Ps_y_rep + s_points_rep * Vy_rep).view(-1, 1)
    T_coords = T_v_rep_2d.view(-1, 1)
    
    # NN入力作成 (N_data * beam_res, 3)
    P_input = torch.cat([X_coords, Y_coords, T_coords], dim=1)
    
    # 推論
    P_nn_path = model(P_input)
    
    # (N_data, beam_res) に戻して積分
    P_nn_path_reshaped = P_nn_path.view(N_data, beam_res)
    
    # --- 積分 ---
    ds_step_rep = L_beam / (beam_res - 1)
    
    integral_sum = torch.sum(P_nn_path_reshaped[:, 1:-1], dim=1).unsqueeze(1)
    integral_val = integral_sum + (P_nn_path_reshaped[:, 0].unsqueeze(1) + P_nn_path_reshaped[:, -1].unsqueeze(1)) / 2
    integral_val *= ds_step_rep

    return integral_val / dataset.I_max

def pinn_loss_function(model, X_p, T_v, V_v, L_v_params, dataset, lambda_param=1e4, X_ic=None, beam_res=50):
    """
    PINNの総合損失関数 L(theta) = L_p + lambda * L_v + lambda_ic * L_ic を計算する。
    """
    # --- 物理損失 L_p の計算 (波動方程式の残差) ---
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
    c_squared = model.c**2
    residual_p = (d2p_dx2 + d2p_dy2) - (1.0 / c_squared) * d2p_dt2
    
    # 物理損失 L_p (式(7))
    L_p = torch.mean(residual_p**2)

    
    
    # --- データ損失 L_v の計算 (射影誤差) ---
    
    # ネットワーク出力からの射影積分の計算 h_j[f_NN]
    H_j_p_nn = calculate_projection_integral_from_nn(model, L_v_params, T_v, dataset, beam_res=beam_res)
    
    # データ損失 L_v (式(8))
    L_v = torch.mean((V_v - H_j_p_nn)**2)
    
    # --- 総合損失の計算 (L = L_p + lambda * L_v ) ---
    total_loss = L_p + lambda_param * L_v
    
    return total_loss, L_p, L_v

# ================================================================
# 3. 評価と可視化 (★ 変更: 保存機能の追加)
# ================================================================

def evaluate_and_plot(model, dataset, T_v_full, V_v_full, sl=100, save_dir=None, show_plots=True):
    """
    最終的な結果を評価し、プロットを生成する。
    save_dir が指定された場合、プロットをファイルに保存する。
    show_plots が True の場合、プロットを表示する。
    """
    device = next(model.parameters()).device
    model.eval()
    
    # --- 図4: 射影データ（シノグラム）の可視化 ---
    # データ形状: (N_angles * N_rays, Nt)
    # ここでは "Measurement Index" としてフラットに表示します
    # (角度ごとのブロックが見えるはずです)
    
    N_spatial = dataset.n_angles * dataset.n_rays
    V_v_reshaped = V_v_full.view(N_spatial, dataset.nt).transpose(0, 1).cpu().detach().numpy()
    
    time_labels = np.linspace(0.0, dataset.T, dataset.nt)
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    v_max = 1
    v_min = -v_max
    
    im = ax4.imshow(
        V_v_reshaped,
        aspect='auto',
        cmap='seismic',
        interpolation='nearest', # ブロック構造を見やすくするため nearest
        extent=[0, N_spatial, time_labels.max(), time_labels.min()],
        vmin=v_min, 
        vmax=v_max
    )
    
    ax4.set_title(f'Projection Data (Sinogram) - {dataset.n_angles} Angles x {dataset.n_rays} Rays')
    ax4.set_xlabel('Measurement Index (Angle x Ray Offset)')
    ax4.set_ylabel('Time (t)')
    
    # 角度ごとの境界線を引く
    for i in range(1, dataset.n_angles):
        ax4.axvline(x=i * dataset.n_rays, color='black', linestyle='--', alpha=0.3)
        
    plt.colorbar(im, ax=ax4, label='Line Integral Value (Normalized)')
    
    if save_dir:
        fig4.savefig(os.path.join(save_dir, "fig4_synogram.png"))

    
    # --- 定量的誤差（NMSE, CS）の時系列計算 ---
    # (変更なし)
    time_points_val = np.linspace(0.0, dataset.T, 20) 
    nmse_series = []
    cs_series = []
    
    with torch.no_grad():
        for t_val in time_points_val:
            X_time, P_true_time, _ = dataset.generate_validation_data(sl, sl, 1, frames=[t_val])
            
            P_est_time = model(X_time.to(device)).cpu().detach()
            P_true_time = P_true_time.cpu()
            
            def calculate_nmse(p_true, p_est):
                p_range = p_true.max() - p_true.min()
                if p_range.item() < 1e-6: 
                    if dataset.I_max > 1e-6:
                        return torch.sqrt(torch.mean((p_true - p_est)**2)).item() / dataset.I_max
                    else:
                        return torch.sqrt(torch.mean((p_true - p_est)**2)).item()
                        
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
    # (前回の修正を適用済み)
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 5))
    
    ax3[0].plot(time_points_val, nmse_series, label='PINN NMSE', marker='o')
    ax3[0].set_title('Normalized Mean Square Error (NMSE) over Time')
    ax3[0].set_xlabel('Time (t)')
    ax3[0].set_xlim(2*11/26,2*18/26)
    ax3[0].set_ylabel('NMSE')
    ax3[0].set_ylim(0, 0.4)#max(np.mean(nmse_series) * 3, 1.0))
    ax3[0].grid(True)
    
    ax3[1].plot(time_points_val, cs_series, label='PINN CS', marker='o', color='red')
    ax3[1].set_title('Cosine Similarity (CS) over Time')
    ax3[1].set_xlabel('Time (t)')
    ax3[1].set_xlim(2*11/26,2*18/26)
    ax3[1].set_ylabel('CS')
    ax3[1].set_ylim(min(np.min(cs_series) - 0.1, 0), 1.05)
    ax3[1].grid(True)
    
    fig3.suptitle("Quantitative Error Metrics (Fig. 3 Emulation)", fontsize=14)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_dir:
        fig3.savefig(os.path.join(save_dir, "fig3_metrics.png"))

    
    # --- 図2: 2D波面スナップショットのプロット ---
    # (★ ここからが修正箇所)
    
    frames = np.linspace(dataset.T * 11/26, dataset.T * 18/26, 4).tolist()
    X_val, P_val_true, frames = dataset.generate_validation_data(sl, sl, len(frames), frames=frames)
    
    with torch.no_grad():
        P_val_est = model(X_val.to(device)).cpu().detach()
    
    P_val_true_reshaped = P_val_true.view(len(frames), sl, sl).numpy()
    P_val_est_reshaped = P_val_est.view(len(frames), sl, sl).numpy()
    
    v_max_true = np.abs(P_val_true_reshaped).max()
    v_max_est = np.abs(P_val_est_reshaped).max()
    v_max_plot = 1#max(v_max_true, v_max_est)
    v_min_plot = -v_max_plot
    if v_max_plot == 0: v_max_plot = 1.0; v_min_plot = -1.0
    
    # ★ 修正: layout='constrained' を指定
    fig2, axes = plt.subplots(len(frames), 2, figsize=(8, 4 * len(frames)), layout='constrained')
    
    im_est = None 
    
    for i in range(len(frames)):
        
        ax_true = axes[i, 0]
        im_true = ax_true.imshow(
            P_val_true_reshaped[i], 
            cmap='seismic', 
            vmin=v_min_plot, 
            vmax=v_max_plot, 
            extent=[-dataset.L/2, dataset.L/2, -dataset.L/2, dataset.L/2],
            origin='lower'
        )
        ax_true.set_title(f"Reference (t={frames[i]:.2f})")
        ax_true.set_aspect('equal')
        ax_true.set_xlabel('X')
        ax_true.set_ylabel('Y')
        
        ax_est = axes[i, 1]
        im_est = ax_est.imshow(
            P_val_est_reshaped[i], 
            cmap='seismic', 
            vmin=v_min_plot, 
            vmax=v_max_plot, 
            extent=[-dataset.L/2, dataset.L/2, -dataset.L/2, dataset.L/2],
            origin='lower'
        )
        ax_est.set_title(f"PINN Estimate (t={frames[i]:.2f})")
        ax_est.set_aspect('equal')
        ax_est.set_xlabel('X')
        
        # ★ 修正: カラーバーをループ内に戻し、サイズ調整
        # (im_est を使うことで、最後のimshowを参照)
        fig2.colorbar(im_est, ax=axes[i, :], shrink=0.8, pad=0.02, label='Pressure')

    fig2.suptitle("Pressure Field Snapshots (Fig. 2 Emulation)", fontsize=14)
    
    # ★ 修正: tight_layout() は layout='constrained' を指定したため不要
    # fig2.tight_layout(rect=[0, 0.03, 1, 0.95]) # ← この行を削除

    if save_dir:
        fig2.savefig(os.path.join(save_dir, "fig2_snapshots.png"))

    # --- プロットの表示/非表示 ---
    if show_plots:
        plt.show()
    else:
        # 保存時はメモリ解放のためにプロットを閉じる
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)


# ================================================================
# 4. ★ 新規: トレーニングとロギング関数
# ================================================================

def plot_losses(loss_history, save_path=None, show_plots=True):
    """
    トレーニング中の損失履歴をプロットする。
    loss_history は 1000 エポックごとに記録されていることを前提とする。
    """
    epochs_logged = len(loss_history['total'])
    if epochs_logged == 0:
        print("損失履歴がありません。プロットをスキップします。")
        return

    # X軸のラベルをエポック数（1000, 2000...）に合わせる
    epochs = np.arange(1, epochs_logged + 1) * 1000
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss_history['total'], label=f'Total Loss (Final: {loss_history["total"][-1]:.4e})', color='black', marker='o', markersize=3)
    ax.plot(epochs, loss_history['L_p'], label=f'L_p (Physics) (Final: {loss_history["L_p"][-1]:.4e})', linestyle='--', marker='x', markersize=3)
    ax.plot(epochs, loss_history['L_v'], label=f'L_v (Data) (Final: {loss_history["L_v"][-1]:.4e})', linestyle='--', marker='s', markersize=3)
    
    ax.set_yscale('log')
    ax.set_title('Training Loss History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Log Scale)')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
        
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def train(model, dataset, params, device, T_v_full, V_v_full, L_v_params_full):
    """
    PINNモデルのトレーニングループを実行する。
    """
    
    # パラメータの展開
    N_EPOCHS = params.get("N_EPOCHS", 10000)
    N_COLLOCATION = params.get("N_COLLOCATION", 100)
    N_IC = params.get("N_IC", 1000)
    LR = params.get("LR", 1e-4)
    LAMBDA = params.get("LAMBDA", 1e5)
    BEAM_RES = params.get("beam_res", 50)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 損失履歴を保存する辞書
    loss_history = {
        'total': [],
        'L_p': [],
        'L_v': [],
    }
    
    print(f"\n--- PINN トレーニング開始 (λ={LAMBDA:.1e}, Epochs={N_EPOCHS}) ---")
    
    start_time = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        X_p = dataset.generate_pinn_collocation_points(N_COLLOCATION).to(device)
        X_ic = None
        
        total_loss, L_p, L_v = pinn_loss_function(
            model, 
            X_p, 
            T_v_full, 
            V_v_full, 
            L_v_params_full, 
            dataset, 
            lambda_param=LAMBDA,
            X_ic=X_ic,
            beam_res=BEAM_RES
        )
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{N_EPOCHS} | Total Loss: {total_loss.item():.6e} | "
                  f"L_p: {L_p.item():.6e} | L_v: {L_v.item():.6e} | Time: {elapsed:.2f}s")
            
            # 損失履歴の記録
            loss_history['total'].append(total_loss.item())
            loss_history['L_p'].append(L_p.item())
            loss_history['L_v'].append(L_v.item())
            
            start_time = time.time()
    
    print("\n--- トレーニング完了 ---")
    return model, loss_history


# ================================================================
# 5. ★ 新規: メイン実行ブロック (実験管理)
# ================================================================

def main(args):
    """
    メインの実行関数。パラメータの管理、ロギング、トレーニング、評価を行う。
    """
    
    # --- デフォルトの変数プリセット ---
    default_params = {
        "experiment": args.experiment,
        "preset": args.preset,
        "N_EPOCHS": 100000,
        "N_COLLOCATION": 100,
        "N_IC": 1000,
        "LR": 1e-3,
        "LAMBDA": 1e4,
        "L": 1.0,
        "T": 2.0,
        "c": 1.2,
        "nl": 20,
        "nt": 26,
        "frequency": 2.0,
        "snr_db": 20.0,
        "beam_res": 50,
        "n_angles": 18 # ★ 追加: 角度の分割数 (例: 10度刻みなら18)
    }
    
    

    print("射影データ生成中...")
    
    log_dir = f"log/{args.experiment}/{args.preset}"
    preset_path = os.path.join(log_dir, "params.json")
    
    # --- プリセットの読み込み (要求 3) ---
    if args.load_preset:
        try:
            with open(preset_path, 'r') as f:
                params = json.load(f)
            print(f"パラメータを {preset_path} から読み込みました。")
            # コマンドライン引数を上書き（読み込んだファイルの設定を優先）
            params['experiment'] = args.experiment
            params['preset'] = args.preset
        except FileNotFoundError:
            print(f"警告: {preset_path} が見つかりません。デフォルトのパラメータを使用します。")
            params = default_params
    else:
        params = default_params
        print("デフォルトのパラメータを使用します。")

    # --- ロギング設定 (要求 1) ---
    save_logs_input = input("この実行のログ（モデル、プロット）を保存しますか？ (y/n): ").strip().lower()
    save_dir = None
    show_plots = True
    
    if save_logs_input == 'y':
        show_plots = False # 保存時はプロットを自動表示しない
        
        # プリセットフォルダの作成
        os.makedirs(log_dir, exist_ok=True)
        
        # --- バージョン管理 (要求 1) ---
        existing_versions = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d.startswith('v')]
        if not existing_versions:
            new_version = 1
        else:
            # v1, v2, v10 などを数値としてソート
            max_v = max([int(v.replace('v', '')) for v in existing_versions])
            new_version = max_v + 1
            
        save_dir = os.path.join(log_dir, f"v{new_version}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"ログを {save_dir} に保存します。")

        # --- 変数プリセットの保存 (要求 3) ---
        try:
            with open(preset_path, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"現在のパラメータを {preset_path} に保存しました。")
        except Exception as e:
            print(f"エラー: パラメータファイル ({preset_path}) の保存に失敗しました: {e}")

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # --- データセットとモデルの準備 ---
    print("データセット準備中...")
    # ★ 修正: n_angles を渡す
    dataset = AcousticTomographyDataset(
        L=params['L'], T=params['T'], c=params['c'], 
        nl=params['nl'], n_angles=params.get('n_angles', 18), # getで安全に取得
        nt=params['nt'], frequency=params['frequency'], 
        snr_db=params['snr_db']
    )
    
    print("射影データ生成中...")
    T_v, V_v, L_v_params = dataset.generate_projection_data(beam_res=params['beam_res'])
    T_v_full = T_v.to(device)
    V_v_full = V_v.to(device)
    L_v_params_full = L_v_params.to(device)
    print(f"射影データポイント数: {V_v_full.shape[0]}")
    
    model = PINNModel(c=dataset.c).to(device)
    
    # --- トレーニング実行 ---
    model, loss_history = train(
        model, 
        dataset, 
        params, 
        device, 
        T_v_full, 
        V_v_full, 
        L_v_params_full
    )
    
    # ジオメトリ画像を保存する (ログ保存モードならフォルダに、そうでなければ表示)
    geo_save_path = os.path.join(save_dir, "geometry_check.png") if save_dir else None
    dataset.plot_geometry(save_path=geo_save_path)
    # ------------------------------------------
    
    # --- 評価と保存 (要求 2) ---
    if save_dir:
        print(f"結果を {save_dir} に保存中...")
        
        # 1. モデルの保存
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
        
        # 2. 損失プロットの保存
        plot_losses(loss_history, save_path=os.path.join(save_dir, "loss_plot.png"), show_plots=False)
        
        # 3. 評価プロット（図2, 3, 4）の保存
        evaluate_and_plot(model, dataset, T_v_full, V_v_full, sl=100, save_dir=save_dir, show_plots=False)
        
        print("保存が完了しました。")
    
    else:
        # 保存しない場合は、結果を表示する
        print("結果を表示します (保存はしません)...")
        plot_losses(loss_history, show_plots=True)
        evaluate_and_plot(model, dataset, T_v_full, V_v_full, sl=100, show_plots=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PINN for 2D Acoustic Tomography with Experiment Management")
    
    parser.add_argument("--experiment", 
                        type=str, 
                        default="ParaMltiAddWave1Tanh", 
                        help="Name of the experiment (e.g., 'source_location_test')")
    
    parser.add_argument("--preset", 
                        type=str, 
                        default="default_preset", 
                        help="Name of the parameter preset (e.g., 'high_lambda_low_lr')")
    
    parser.add_argument("--load_preset", 
                        action="store_true", 
                        help="Load parameters from the specified experiment/preset directory instead of using defaults.")
    
    args = parser.parse_args()
    
    main(args)