import torch
import numpy as np
import matplotlib.pyplot as plt

class AcousticParallelBeamTomographyDataset:
    """
    PINNトレーニング用のデータ生成クラス。
    平行ビーム・回転スキャン (Parallel Beam Tomography) に変更
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
        return P1 + P2

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


class AcousticParallelBeamTomographyDataset_AddWave1:
    """
    PINNトレーニング用のデータ生成クラス。
    平行ビーム・回転スキャン (Parallel Beam Tomography) に変更
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



class AcousticParallelBeamTomographyDataset_RefineLP:
    """
    PINNトレーニング用のデータ生成クラス。
    平行ビーム・回転スキャン (Parallel Beam Tomography) に変更
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
        return P1 + P2

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

    def generate_pinn_collocation_points(self, N_p, scale=1.0):
        """
        物理損失評価用のコロケーションポイントを生成する。
        
        Args:
            N_p (int): 生成する点の数
            scale (float): 領域の拡大倍率。
                           1.0 = ROI内部 [-L/2, L/2]
                           2.0 = 2倍の範囲 [-L, L] (データパスを含む範囲)
                           3.0 = さらに広い外挿領域
        """
        X_p = torch.rand(N_p, 3)
        
        # 領域幅の計算
        current_width = self.L * scale
        
        # 中心(0,0)から current_width の範囲に展開
        # [0, 1] -> [-current_width/2, current_width/2]
        X_p[:, :2] = X_p[:, :2] * current_width - current_width/2
        
        # 時間方向は変更なし [0, T]
        X_p[:, 2] = X_p[:, :2].new_zeros(N_p).uniform_(0, self.T) # 時間も一様乱数で再生成推奨(元のコードはrand流用でOKですが明示的に)
        # 元のコードの流儀に合わせるなら以下でもOK
        # X_p[:, 2] = X_p[:, 2] * self.T
        
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



class AcousticParallelBeamTomographyDataset_RefineLP_AddWave1:
    """
    PINNトレーニング用のデータ生成クラス。
    平行ビーム・回転スキャン (Parallel Beam Tomography) に変更
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

    def generate_pinn_collocation_points(self, N_p, scale=1.0):
        """
        物理損失評価用のコロケーションポイントを生成する。
        
        Args:
            N_p (int): 生成する点の数
            scale (float): 領域の拡大倍率。
                           1.0 = ROI内部 [-L/2, L/2]
                           2.0 = 2倍の範囲 [-L, L] (データパスを含む範囲)
                           3.0 = さらに広い外挿領域
        """
        X_p = torch.rand(N_p, 3)
        
        # 領域幅の計算
        current_width = self.L * scale
        
        # 中心(0,0)から current_width の範囲に展開
        # [0, 1] -> [-current_width/2, current_width/2]
        X_p[:, :2] = X_p[:, :2] * current_width - current_width/2
        
        # 時間方向は変更なし [0, T]
        X_p[:, 2] = X_p[:, :2].new_zeros(N_p).uniform_(0, self.T) # 時間も一様乱数で再生成推奨(元のコードはrand流用でOKですが明示的に)
        # 元のコードの流儀に合わせるなら以下でもOK
        # X_p[:, 2] = X_p[:, 2] * self.T
        
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