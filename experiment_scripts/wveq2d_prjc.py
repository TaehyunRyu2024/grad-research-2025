import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# PyTorchとNumPyのシードを設定して再現性を確保
torch.manual_seed(42)
np.random.seed(42)

# ================================================================
# 1. データセットクラス (実験設定とデータ生成)
# ================================================================

class AcousticTomographyDataset:
    """
    音場の実験設定を定義し、真の測定データを生成するクラス。
    論文のセクション4の2D点音源の実験に基づいています。
    """
    def __init__(self, L=1.0, T=2.0, c=1.0, nl=20, nt=26, frequency=1, snr_db=20.0, pulse_width=0.1):
        # 物理パラメータ (無次元化を想定)
        self.L = L # 空間ドメインの半長 ([-L/2, L/2]を使用)
        self.T = T # 時間ドメインの最大値
        self.c = c # 音速 (無次元化された値)
        self.freq = frequency # 波のピーク周波数

        # 測定/サンプリングパラメータ
        self.nl = nl # プローブビームの数 (論文と同じ20)
        self.nt = nt # 時間サンプルの数 (論文と同じ26)
        self.snr_db = snr_db # ノイズSNR (20 dB)
        
        # 音源位置 (論文に基づきスケーリング)
        self.src1 = torch.tensor([1.5 * L, 0.0])
        self.src2 = torch.tensor([-L, L])
        
        # --- ファンビーム ジオメトリ ---
        self.source_x = 0.0
        self.source_y = -self.L / 2 
        self.detector_y = self.L / 2
        self.detector_x_coords = torch.linspace(-self.L / 2, self.L / 2, self.nl).float()
        
        # 正規化のための最大振幅を推定
        self.P_max = self._calculate_max_pressure()
        if self.P_max < 1e-4: self.P_max = 1.0
        self.I_max = self.L * self.P_max * 1.5

    def _free_space_green_function_2d(self, R, t):
        """
        2D自由空間における点音源の応答。
        波形をリッカー波（Ricker wavelet）に変更。
        """
        R = torch.clamp(R, min=1e-6)
        time_delay = R / self.c
        amplitude = 1.0 / torch.sqrt(R) 
        
        t_local = t - time_delay
        
        # --- リッカー波（Mexican Hat Wavelet）の実装 ---
        # f(t') = (1 - 2 * (pi * f * t')^2) * exp(-(pi * f * t')^2)
        pf = np.pi * self.freq
        pf2 = pf**2
        
        ricker_term = (1 - 2 * pf2 * t_local**2) * torch.exp(-pf2 * t_local**2)
        
        return amplitude * ricker_term/5

    def true_pressure(self, X):
        """任意の時空間座標 (x, y, t) における真の音響圧力場を計算。"""
        r = X[:, :2]
        t = X[:, 2].unsqueeze(1)
        
        R1 = torch.linalg.norm(r - self.src1.to(r.device), dim=1).unsqueeze(1)
        P1 = self._free_space_green_function_2d(R1, t)
        
        R2 = torch.linalg.norm(r - self.src2.to(r.device), dim=1).unsqueeze(1)
        P2 = self._free_space_green_function_2d(R2, t)
        
        return P1 + P2

    def _calculate_max_pressure(self, N=10000):
        """真の音場の最大振幅を推定。"""
        X_test = torch.rand(N, 3)
        X_test[:, :2] = X_test[:, :2] * 3 * self.L - 1.5 * self.L # 音源を含む広い範囲でサンプリング
        X_test[:, 2] = X_test[:, 2] * self.T
        with torch.no_grad():
            P_test = self.true_pressure(X_test)
        return P_test.abs().max().item()

    def generate_projection_data(self, beam_res=50):
        """射影データ V_v と、それに対応するパラメータを生成。"""
        T_v = torch.linspace(0.0, self.T, self.nt).unsqueeze(1) 
        N_data = self.nl * self.nt
        s_points = torch.linspace(0.0, 1.0, beam_res).unsqueeze(1)
        
        V_v_true = torch.zeros(N_data, 1) 
        L_v_params = torch.zeros(N_data, 2)
        
        idx = 0
        Ps = torch.tensor([self.source_x, self.source_y])
        
        for i in range(self.nl):
            det_x = self.detector_x_coords[i]
            L_v_params[idx:idx + self.nt, 0] = det_x
            Pd = torch.tensor([det_x.item(), self.detector_y])
            V = Pd - Ps
            L_beam = torch.linalg.norm(V).item()
            V_norm = V / L_beam
            s_path = s_points * L_beam 
            X_coords = Ps[0] + s_path * V_norm[0].item()
            Y_coords = Ps[1] + s_path * V_norm[1].item()
            ds_step = L_beam / (beam_res - 1)
            
            for j in range(self.nt):
                t = T_v[j]
                T_coords = torch.full_like(X_coords, t.item())
                P_input = torch.cat([X_coords, Y_coords, T_coords], dim=1)
                P_true_path = self.true_pressure(P_input)
                
                integral_val = torch.sum(P_true_path[1:-1]).item() 
                integral_val += (P_true_path[0].item() + P_true_path[-1].item()) / 2
                integral_val *= ds_step
                V_v_true[idx, 0] = integral_val
                idx += 1
                
        P_rms = torch.sqrt(torch.mean(V_v_true**2))
        noise_power = P_rms / (10**(self.snr_db / 20.0))
        noise = torch.randn_like(V_v_true) * noise_power.item()
        
        V_v_true_scaled = V_v_true / self.I_max
        V_v_noisy_scaled = (V_v_true + noise) / self.I_max
        T_v_all = T_v.repeat_interleave(self.nl, dim=0).view(-1, 1)
        
        return T_v_all, V_v_noisy_scaled, V_v_true_scaled
        
    def generate_wave_field_data(self, sl_x=100, sl_y=100, n_frames=13):
        """可視化のための2D波面データを生成する。"""
        x = torch.linspace(-self.L / 2, self.L / 2, sl_x)
        y = torch.linspace(-self.L / 2, self.L / 2, sl_y)
        frames = torch.linspace(0.0, self.T, n_frames)
        
        XX, YY = torch.meshgrid(x, y, indexing='xy')
        
        wave_data = torch.zeros(n_frames, sl_y, sl_x)

        for i, t in enumerate(frames):
            print(f"  Generating frame {i+1}/{n_frames} (t={t:.2f}s)...")
            TT = torch.full_like(XX, t.item())
            X_input = torch.stack([XX.flatten(), YY.flatten(), TT.flatten()], dim=1)
            with torch.no_grad():
                pressure = self.true_pressure(X_input)
            wave_data[i] = pressure.view(sl_y, sl_x)
            
        return wave_data, frames.numpy()


# ================================================================
# 2. 可視化関数
# ================================================================

def visualize_synogram_comparison(dataset, V_v_noisy, V_v_true):
    """
    ノイズありとノイズなしの射影データ（シノグラム）を比較して可視化する。
    """
    V_v_noisy_reshaped = V_v_noisy.view(dataset.nl, dataset.nt).transpose(0, 1).cpu().numpy()
    V_v_true_reshaped = V_v_true.view(dataset.nl, dataset.nt).transpose(0, 1).cpu().numpy()
    
    time_labels = np.linspace(0.0, dataset.T, dataset.nt)
    detector_x_labels = dataset.detector_x_coords.cpu().numpy()
    
    # 共通のカラースケールを設定
    v_max = max(abs(V_v_noisy_reshaped.min()), abs(V_v_noisy_reshaped.max()), abs(V_v_true_reshaped.min()), abs(V_v_true_reshaped.max()))
    v_min = -v_max

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    # --- 左側: ノイズなし (True Signal) ---
    im1 = ax1.imshow(
        V_v_true_reshaped,
        aspect='auto', cmap='seismic', interpolation='bilinear',
        extent=[detector_x_labels.min(), detector_x_labels.max(), time_labels.max(), time_labels.min()],
        vmin=v_min, vmax=v_max
    )
    ax1.set_title('True Signal (Noiseless)')
    ax1.set_xlabel(f'Detector X Coordinate')
    ax1.set_ylabel('Time (t)')
    
    # --- 右側: ノイズあり (Simulated Measurement) ---
    im2 = ax2.imshow(
        V_v_noisy_reshaped,
        aspect='auto', cmap='seismic', interpolation='bilinear',
        extent=[detector_x_labels.min(), detector_x_labels.max(), time_labels.max(), time_labels.min()],
        vmin=v_min, vmax=v_max
    )
    ax2.set_title('Simulated Measurement (with Noise)')
    ax2.set_xlabel(f'Detector X Coordinate')
    ax2.set_ylabel('Time (t)')

    fig.colorbar(im2, ax=[ax1, ax2], label='Line Integral Value (Normalized)')
    fig.suptitle('Comparison of Projection Data', fontsize=16)
    plt.show()

def visualize_wave_propagation(dataset, wave_data, frames):
    """
    2D波面の時間伝播を可視化する。
    音源の位置と計算領域を明示する。
    """
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    
    v_min = -1#wave_data.min()
    v_max = 1#wave_data.max()
    
    for i, ax in enumerate(axes.flat):
        if i < len(frames):
            im = ax.imshow(
                wave_data[i],
                cmap='seismic',
                vmin=v_min,
                vmax=v_max,
                extent=[-dataset.L / 2, dataset.L / 2, -dataset.L / 2, dataset.L / 2],
                origin='lower'
            )
            ax.set_title(f'Time: {frames[i]:.2f}s')
            
            rect = patches.Rectangle(
                (-dataset.L/2, -dataset.L/2), dataset.L, dataset.L, 
                linewidth=2, edgecolor='g', facecolor='none', linestyle='--', 
                label='Calculation Domain' if i == 0 else ""
            )
            ax.add_patch(rect)
            
            ax.plot(dataset.src1[0], dataset.src1[1], 'r*', markersize=15, label='Source 1' if i == 0 else "")
            ax.plot(dataset.src2[0], dataset.src2[1], 'b*', markersize=15, label='Source 2' if i == 0 else "")
            
            ax.set_xlim(-dataset.L, 1.6 * dataset.L)
            ax.set_ylim(-dataset.L, 1.6 * dataset.L)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            if i == 0:
                ax.legend()
        else:
            ax.axis('off')

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Acoustic Pressure')
    fig.suptitle('Wave Field Propagation Snapshots', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_projection_path(dataset, beam_index=0, t=0.0):
    """
    特定のビーム（beam_index）の線積分パスをt=0.0の波面データ上に可視化する。
    """
    print(f"\n--- Visualizing Projection Path for Beam {beam_index} at t={t:.2f}s ---")
    
    # --- 1. 波面データの準備 ---
    sl_x, sl_y = 100, 100
    x = torch.linspace(-dataset.L / 2, dataset.L / 2, sl_x)
    y = torch.linspace(-dataset.L / 2, dataset.L / 2, sl_y)
    XX, YY = torch.meshgrid(x, y, indexing='xy')
    
    # t=0.0の音場を計算
    TT = torch.full_like(XX, t)
    X_input_wave = torch.stack([XX.flatten(), YY.flatten(), TT.flatten()], dim=1)
    with torch.no_grad():
        pressure_data = dataset.true_pressure(X_input_wave)
    wave_data = pressure_data.view(sl_y, sl_x)
    
    # --- 2. ビームパスの座標計算 ---
    beam_res = 50 
    s_points = torch.linspace(0.0, 1.0, beam_res).unsqueeze(1)
    Ps = torch.tensor([dataset.source_x, dataset.source_y])
    
    # ターゲットビーム（インデックス i=0）の検出点座標
    det_x = dataset.detector_x_coords[beam_index]
    Pd = torch.tensor([det_x.item(), dataset.detector_y])
    
    # ビームベクトルと長さ
    V = Pd - Ps
    L_beam = torch.linalg.norm(V).item()
    V_norm = V / L_beam
    
    # 積分パス上の X, Y 座標
    s_path = s_points * L_beam 
    X_coords = Ps[0] + s_path * V_norm[0].item()
    Y_coords = Ps[1] + s_path * V_norm[1].item()
    
    # --- 3. プロット ---
    
    fig, ax = plt.subplots(figsize=(10, 10))
    v_max = wave_data.abs().max()
    
    # t=0.0の波面をimshowで表示
    im = ax.imshow(
        wave_data,
        cmap='seismic',
        vmin=-v_max,
        vmax=v_max,
        extent=[-dataset.L / 2, dataset.L / 2, -dataset.L / 2, dataset.L / 2],
        origin='lower'
    )
    
    # ビームの積分パス上のサンプリング点を重ねてプロット
    ax.scatter(X_coords.numpy(), Y_coords.numpy(), c='red', s=30, label=f'Sampling Points (Beam {beam_index})', zorder=3)
    # ビームの始点（発散点）と終点（検出点）をマーク
    ax.plot(Ps[0], Ps[1], 'go', markersize=10, label='Source (Ps)', zorder=4)
    ax.plot(Pd[0], Pd[1], 'bo', markersize=10, label='Detector (Pd)', zorder=4)
    
    # 計算領域の枠線
    rect = patches.Rectangle(
        (-dataset.L/2, -dataset.L/2), dataset.L, dataset.L, 
        linewidth=2, edgecolor='g', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)
    
    # 音源位置
    ax.plot(dataset.src1[0], dataset.src1[1], 'r*', markersize=15, label='Acoustic Source 1')
    ax.plot(dataset.src2[0], dataset.src2[1], 'b*', markersize=15, label='Acoustic Source 2')
    
    ax.set_title(f'Projection Path Visualization at t={t:.2f}s')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(-dataset.L, 1.6 * dataset.L)
    ax.set_ylim(-dataset.L, 1.6 * dataset.L)
    ax.legend()
    fig.colorbar(im, ax=ax, label='Acoustic Pressure')
    plt.show()

# ================================================================
# 3. メイン実行ブロック
# ================================================================

if __name__ == '__main__':
    print("論文の実験設定に基づいて測定データを生成します...")
    
    dataset = AcousticTomographyDataset(
        L=1.0, 
        T=2.0, 
        c=1.5, 
        nl=20, 
        nt=26, 
        frequency=2, 
        snr_db=20.0
    )
    
    # --- 射影データの生成と可視化 ---
    print("\n--- Generating Projection Data (Synogram) ---")
    start_time = time.time()
    _, V_v_noisy, V_v_true = dataset.generate_projection_data(beam_res=50)
    end_time = time.time()
    
    print(f"データ生成完了。({end_time - start_time:.2f}秒)")
    visualize_synogram_comparison(dataset, V_v_noisy, V_v_true)
    
    # --- 線積分パスの可視化 ---
    visualize_projection_path(dataset, beam_index=0, t=0.33)
    
    # --- 2D波面データの生成と可視化 ---
    print("\n--- Generating Wave Field Snapshots ---")
    start_time = time.time()
    wave_data, frames = dataset.generate_wave_field_data(sl_x=100, sl_y=100, n_frames=13)
    end_time = time.time()
    print(f"波面データ生成完了。({end_time - start_time:.2f}秒)")
    visualize_wave_propagation(dataset, wave_data, frames)
