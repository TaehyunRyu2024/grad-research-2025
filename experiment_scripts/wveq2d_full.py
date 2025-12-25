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
        
        # 音源位置 (論文に基づきスケーリング - 計算領域外に配置)
        self.src1 = torch.tensor([1.5 * L, 0.0])
        self.src2 = torch.tensor([-L, L])
        
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
        X_test = torch.rand(N, 3) * self.L - self.L/2
        X_test[:, 2] = X_test[:, 2] * (self.T / self.L) + self.T/2
        with torch.no_grad():
            P_test = self.true_pressure(X_test)
        return P_test.abs().max().item()

    def generate_projection_data(self, beam_res=50):
        """
        射影データ V_v と、それに対応するパラメータを生成する。
        """
        # T_v (nt, 1) 形状
        T_v = torch.linspace(0.0, self.T, self.nt).unsqueeze(1) 
        N_data = self.nl * self.nt
        s_points = torch.linspace(0.0, 1.0, beam_res).unsqueeze(1)
        
        V_v_true = torch.zeros(N_data, 1) 
        L_v_params = torch.zeros(N_data, 2)
        
        idx = 0
        Ps = torch.tensor([self.source_x, self.source_y])
        
        # このループ (nl が外側、nt が内側) が「ビーム優先」の順序を決定する
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
                t = T_v[j] # T_v[0], T_v[1], ...
                
                T_coords = torch.full_like(X_coords, t.item())
                P_input = torch.cat([X_coords, Y_coords, T_coords], dim=1)
                P_true_path = self.true_pressure(P_input)
                
                # ... (積分) ...
                integral_val = torch.sum(P_true_path[1:-1]).item() 
                integral_val += (P_true_path[0].item() + P_true_path[-1].item()) / 2
                integral_val *= ds_step
                V_v_true[idx, 0] = integral_val
                idx += 1
        
        # --- (I_max の決定) ---
        actual_I_max = torch.max(torch.abs(V_v_true))
        if actual_I_max.item() < 1e-6:
            print("警告: 積分データの最大振幅がほぼゼロです。I_max=1.0 を使用します。")
            self.I_max = 1.0
        else:
            self.I_max = actual_I_max.item() 
        print(f"データセットの I_max (最大積分振幅) を {self.I_max:.4e} に設定しました。")

        # --- (ノイズ追加) ---
        P_rms = torch.sqrt(torch.mean(V_v_true**2))
        noise_power = P_rms / (10**(self.snr_db / 20.0))
        noise = torch.randn_like(V_v_true) * noise_power.item()
        
        V_v_true_scaled = V_v_true / self.I_max
        V_v_scaled = (V_v_true + noise) / self.I_max 
        
        # --- 修正: T_v_all の生成ロジック ---
        #
        # 誤 (時間優先): T_v.repeat_interleave(self.nl, dim=0).view(-1, 1)
        #    [T0, T0, T1, T1] (nt=2, nl=2 の場合)
        #
        # 正 (ビーム優先): T_v.repeat(self.nl, 1)
        #    [T0, T1, T0, T1] (nt=2, nl=2 の場合)
        #
        T_v_all = T_v.repeat(self.nl, 1) # (N_data, 1) 形状になる
        
        # --- 修正: return 文に L_v_params を含める (以前の会話で欠落していた) ---
        return T_v_all, V_v_scaled, L_v_params

    def generate_pinn_collocation_points(self, N_p):
        """
        物理損失 L_p のためのランダムなコロケーションポイント X_p を生成する。
        """
        X_p = torch.rand(N_p, 3)
        
        # x, y 座標を [-L/2, L/2] にスケーリング
        X_p[:, :2] = X_p[:, :2] * self.L - self.L/2
        
        # t 座標を [0, T] にスケーリング
        X_p[:, 2] = X_p[:, 2] * self.T
        
        return X_p

    def generate_initial_collocation_points(self, N_ic):
        """
        初期条件 L_ic のためのコロケーションポイント X_ic を生成する。
        時間 t=0 の空間座標のみ。
        """
        X_ic = torch.rand(N_ic, 3)
        
        # x, y 座標を [-L/2, L/2] にスケーリング
        X_ic[:, :2] = X_ic[:, :2] * self.L - self.L/2
        
        # 時間 t=0 に設定
        X_ic[:, 2] = 0.0
        
        return X_ic


    def generate_validation_data(self, sl_x, sl_y, n_frames, frames=None):
        """
        検証用の時空間グリッドと真の圧力場を生成する。
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
    """
    def __init__(self, c=1.0):
        super(PINNModel, self).__init__()
        self.c = c
        
        # ネットワーク層の定義 (3 -> 64 -> 64 -> 64 -> 1)
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            #Sine(),
            nn.Linear(64, 64),
            nn.Tanh(),
            #Sine(),
            nn.Linear(64, 64),
            nn.Tanh(),
            #Sine(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

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

def pinn_loss_function(model, X_p, T_v, V_v, L_v_params, dataset, lambda_param=1e4, X_ic=None, lambda_ic=1.0):
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

    # --- 初期条件損失 L_ic の計算 ---
    #L_ic = torch.tensor(0.0).to(X_p.device)
    #if X_ic is not None:
        #X_ic.requires_grad_(True) 
        #p_ic_nn = model(X_ic)
        
        # IC1: 初期圧力 p(r, 0) = 0
        #L_ic1 = torch.mean(p_ic_nn**2)
        
        # IC2: 初期速度 dp/dt(r, 0) = 0
        #dp_dt_ic = grad(p_ic_nn, X_ic, grad_outputs=torch.ones_like(p_ic_nn), create_graph=True)[0][:, 2].unsqueeze(1)
        #L_ic2 = torch.mean(dp_dt_ic**2)
        
        # IC 総合損失
        #L_ic = L_ic1 + L_ic2
    
    # --- データ損失 L_v の計算 (射影誤差) ---
    
    # ネットワーク出力からの射影積分の計算 h_j[f_NN]
    H_j_p_nn = calculate_projection_integral_from_nn(model, L_v_params, T_v, dataset)
    
    # データ損失 L_v (式(8))
    L_v = torch.mean((V_v - H_j_p_nn)**2)
    
    # --- 総合損失の計算 (L = L_p + lambda * L_v + lambda_ic * L_ic) ---
    total_loss = L_p + lambda_param * L_v# + lambda_ic * L_ic
    
    return total_loss, L_p, L_v, L_ic

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

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
    
    # --- 修正: (nt, nl) で view していたバグを修正 ---
    #
    # 誤: V_v_reshaped = V_v_full.view(dataset.nt, dataset.nl).cpu().numpy()
    #
    # 正: V_v_full は (nl, nt) の順序で格納されているため、
    #    (nl, nt) で view し、その後 (nt, nl) に転置(transpose)する
    
    V_v_reshaped = V_v_full.view(dataset.nl, dataset.nt).transpose(0, 1).cpu().detach().numpy()
    
    # 時間軸のラベル設定
    time_labels = np.linspace(0.0, dataset.T, dataset.nt)
    
    # X軸: ビームインデックスではなく、検出器の実際のX座標を使用
    detector_x_labels = dataset.detector_x_coords.cpu().numpy()
    
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    
    # 共通のカラースケールを設定 (ノイズなしデータがないため、V_v_reshaped のみで設定)
    v_max = np.abs(V_v_reshaped).max()
    v_min = -v_max
    if v_max == 0: v_max = 1.0; v_min = -1.0 # ゼロ除算防止

    # extent: [xmin, xmax, ymin, ymax]。原点 (0, 0) は左上になるように設定
    im = ax4.imshow(
        V_v_reshaped,
        aspect='auto',
        cmap='seismic',
        interpolation='bilinear', # 'none' より 'bilinear' の方がシノグラムは見やすい
        # --- 修正: X軸をビームインデックスから検出器X座標に変更 ---
        extent=[detector_x_labels.min(), detector_x_labels.max(), time_labels.max(), time_labels.min()],
        vmin=v_min, 
        vmax=v_max
    )
    
    ax4.set_title('Measured Projection Data (Synogram) - V_v (Fig. 4)')
    ax4.set_xlabel('Detector X Coordinate') # 'Beam Index' から変更
    ax4.set_ylabel('Time (t)')
    plt.colorbar(im, ax=ax4, label='Line Integral Value (Normalized)')
    plt.show()

    
    # --- 定量的誤差（NMSE, CS）の時系列計算 ---
    time_points_val = np.linspace(0.0, dataset.T, 20) 
    nmse_series = []
    cs_series = []
    
    with torch.no_grad():
        for t_val in time_points_val:
            # t_valでの単一のスナップショットを生成 (評価領域 [-L/2, L/2])
            # (X_val は [x, y, t] の順序)
            X_time, P_true_time, _ = dataset.generate_validation_data(sl, sl, 1, frames=[t_val])
            
            # 予測
            P_est_time = model(X_time.to(device)).cpu().detach()
            P_true_time = P_true_time.cpu()
            
            # メトリック計算 (NMSEとCSのユーティリティ関数をインラインで定義)
            def calculate_nmse(p_true, p_est):
                # 論文に従い、最大値と最小値の差で正規化
                p_range = p_true.max() - p_true.min()
                if p_range.item() < 1e-6: 
                    # t=0.0 付近で P_true がほぼゼロで p_range が小さい場合、NMSEが発散するのを防ぐ
                    # この場合、純粋なRMSEを返すか、固定の最小レンジを使用
                    # I_max ではなく P_max (dataset.P_max) を使いたいが、
                    # dataset.P_max が正確でない可能性があるため、I_max (積分最大値) で代用
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
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 5))
    
    ax3[0].plot(time_points_val, nmse_series, label='PINN NMSE', marker='o')
    ax3[0].set_title('Normalized Mean Square Error (NMSE) over Time')
    ax3[0].set_xlabel('Time (t)')
    ax3[0].set_ylabel('NMSE')
    #ax3[0].set_xlim(1, 2) # t=0 付近の NMSE が発散する場合があるため、xlim を設定
    ax3[0].set_ylim(0, max(np.mean(nmse_series) * 3, 1.0)) # Y軸の上限を設定
    ax3[0].grid(True)
    
    ax3[1].plot(time_points_val, cs_series, label='PINN CS', marker='o', color='red')
    ax3[1].set_title('Cosine Similarity (CS) over Time')
    ax3[1].set_xlabel('Time (t)')
    ax3[1].set_ylabel('CS')
    #ax3[1].set_xlim(1, 2)
    ax3[1].set_ylim(min(np.min(cs_series) - 0.1, 0), 1.05) # Y軸の下限も調整
    ax3[1].grid(True)
    
    fig3.suptitle("Quantitative Error Metrics (Fig. 3 Emulation)", fontsize=14)
    plt.show()

    # --- 図2: 2D波面スナップショットのプロット ---
    
    # 論文の図2に合わせたフレーム
    frames = np.linspace(dataset.T * 0.5, dataset.T * 0.9, 4).tolist()
    
    # (X_val は [x, y, t] の順序)
    X_val, P_val_true, frames = dataset.generate_validation_data(sl, sl, len(frames), frames=frames)
    
    # 推定値を計算
    with torch.no_grad():
        P_val_est = model(X_val.to(device)).cpu().detach()
    
    P_val_true_reshaped = P_val_true.view(len(frames), sl, sl).numpy()
    P_val_est_reshaped = P_val_est.view(len(frames), sl, sl).numpy()
    
    # 全スナップショットの共通カラーマップ範囲を決定
    v_max_true = np.abs(P_val_true_reshaped).max()
    v_max_est = np.abs(P_val_est_reshaped).max()
    v_max_plot = max(v_max_true, v_max_est)
    v_min_plot = -v_max_plot
    if v_max_plot == 0: v_max_plot = 1.0; v_min_plot = -1.0 # ゼロ除算防止
    
    fig2, axes = plt.subplots(len(frames), 2, figsize=(8, 4 * len(frames)))
    
    for i in range(len(frames)):
        
        # 1列目: 参照 (Reference)
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
        
        # 2列目: PINN推定 (Estimate)
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
        
        # カラーバーの追加
        fig2.colorbar(im_est, ax=axes[i, :], shrink=0.6, label='Pressure')

    fig2.suptitle("Pressure Field Snapshots (Fig. 2 Emulation)", fontsize=14)
    plt.tight_layout()
    plt.show()

# ================================================================
# 4. メイン実行ブロック
# ================================================================

if __name__ == '__main__':
    # --- ハイパーパラメータ設定 (論文の記述に基づく) ---
    N_EPOCHS = 10000 # 論文は 5 * 10^4 エポック
    N_COLLOCATION = 100 # L_p ポイント (論文は100)
    N_IC = 1000 # 追加: 初期条件ポイント数
    LR = 1e-4
    LAMBDA = 1e5 # データ損失の重み
    #LAMBDA_IC = 0#1e5 # 追加: 初期条件損失の重み (データ損失より大きくすることが多い)
    
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
    
    # モデルとオプティマイザの初期化
    model = PINNModel(c=dataset.c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # --- トレーニングループ ---
    print(f"\n--- PINN トレーニング開始 (λ={LAMBDA:.1e}, Epochs={N_EPOCHS}) ---")#, λ_ic={LAMBDA_IC:.1e}
    
    start_time = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # L_p のための新しいコロケーションポイントを各エポックでサンプリング
        X_p = dataset.generate_pinn_collocation_points(N_COLLOCATION).to(device)

        # L_ic のための初期条件ポイントをサンプリング
        X_ic = dataset.generate_initial_collocation_points(N_IC).to(device)
        
        # 損失の計算
        total_loss, L_p, L_v, L_ic = pinn_loss_function(
            model, 
            X_p, 
            T_v_full, 
            V_v_full, 
            L_v_params_full, 
            dataset, 
            lambda_param=LAMBDA,
            X_ic=X_ic,
            #lambda_ic=LAMBDA_IC
        )
        
        # バックプロパゲーションと最適化
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{N_EPOCHS} | Total Loss: {total_loss.item():.6e} | "
                  f"L_p: {L_p.item():.6e} | L_v: {L_v.item():.6e} | Time: {elapsed:.2f}s")# | L_ic: {L_ic.item():.6e}
            start_time = time.time() # 1000エポックごとの時間計測をリセット
    
    print("\n--- トレーニング完了 ---")
    
    # --- 最終評価とプロット ---
    # 波面スナップショットと定量的誤差を表示します。
    # 修正: V_v_fullをevaluate_and_plotに渡す
    evaluate_and_plot(model, dataset, T_v_full, V_v_full, sl=100)
