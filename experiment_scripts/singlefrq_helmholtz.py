import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# 既存ライブラリのインポート
from lib.modules import (
    ExperimentManager
)
from lib.dataset import (
    AcousticParallelBeamTomographyDataset as APBTDataset
)
# モデルは SIREN (出力チャネル数を2に変更して複素数に対応させる)
from lib.PINN_model import (
    PINNModel_SIREN_Flexible as PINNModel
)
from lib.train import (
    train_model
)
from lib.evaluation import (
    plot_losses,
    plot_complex_field
    # ヘルムホルツ用のプロット関数は後述の「追加コード」で定義してimportすることを想定
    # ここでは仮に同じファイル内あるいはlibからimportする形をとります
)

# ★ ヘルムホルツ用のLossとPDE定義（後述のコードをlib/lossfunction.pyに追加してください）
# まだlibにない場合は、一時的にこのスクリプト内で定義するか、libに追加した前提でimportします
from lib.lossfunction import (
    HelmholtzEquation2D,  # ★新規
    PINNLoss_Helmholtz,   # ★新規
    ProjectionOperator    # 既存流用 (線積分)
)

# ★ 警告を非表示
import warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")

# PyTorchのシード設定
torch.manual_seed(42)
np.random.seed(42)


# ================================================================
# メイン実行ブロック
# ================================================================

def main(args):
    # --- デフォルトパラメータ ---
    # ヘルムホルツ用に調整
    default_params = {
        "N_EPOCHS": 20000,     # 単一周波数なら少なめでも収束するかも
        "N_COLLOCATION": 2000, # 時間次元がないので空間点を多めに
        "LR": 5e-4,            # 振動的なので少し控えめ
        "LAMBDA": 1.0,         # データ項とPDE項のバランス（要調整）
        "L": 1.0,
        "c": 1.2,
        "nl": 20,
        "nt": 256,             # FFTのために時間ステップ数は多め推奨
        "frequency": 2.0,      # ★ターゲット周波数 (リッカー波のピーク付近)
        "snr_db": 20.0,
        "beam_res": 50,
        "n_angles": 8,
        "target_frq": 2.0
    }
    
    # 1. マネージャーの初期化
    exp_manager = ExperimentManager(args.experiment, args.preset)
    params = exp_manager.load_params(default_params, load_preset=args.load_preset)

    # 2. ログ保存設定
    save_logs_input = input("この実行のログ（モデル、プロット）を保存しますか？ (y/n): ").strip().lower()
    save_dir = exp_manager.setup_log_dir(save_logs=(save_logs_input == 'y'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 3. データセット準備
    # Dataset自体は時間領域のシミュレーションを行う
    dataset = APBTDataset(
        L=params['L'], T=2.0/params['frequency']*10, # 周期の数倍程度の時間を確保
        c=params['c'], 
        nl=params['nl'], n_angles=params.get('n_angles', 8),
        nt=params['nt'], frequency=params['frequency'], snr_db=params['snr_db']
    )
    
    # 時間領域の投影データを取得
    print("時間領域の投影データを生成中...")
    T_v_time, V_v_time, L_v_params = dataset.generate_projection_data(beam_res=params['beam_res'])
    
    # 4. 【重要】FFTによる周波数領域データの抽出
    print(f"ターゲット周波数 {params['target_frq']}Hz 成分を抽出中...")
    
    # V_v_time: (N_rays, nt) or (N_rays, 1, nt) -> datasetの実装によるが、形状確認が必要
    # dataset.pyを見る限り V_v_scaled は (N_data, 1) で flattenされている可能性があるためreshape
    N_spatial = params['n_angles'] * params['nl']
    V_v_reshaped = V_v_time.view(N_spatial, params['nt'])
    
    # FFT実行 (実数 -> 複素数)
    V_v_fft = torch.fft.rfft(V_v_reshaped, dim=-1)
    freqs = torch.fft.rfftfreq(params['nt'], d=(dataset.T / params['nt']))
    
    # ターゲット周波数に最も近いインデックスを探す
    target_idx = (torch.abs(freqs - params['target_frq'])).argmin()
    actual_freq = freqs[target_idx]
    print(f"Target Freq: {params['target_frq']} Hz -> Actual FFT Bin: {actual_freq:.4f} Hz")
    
    # 特定周波数の複素データを抽出 -> (N_spatial, 1)
    V_v_complex = V_v_fft[:, target_idx].unsqueeze(1).to(device)
    
    # L_v_paramsは空間座標(theta, u)だけあればいいので、時間方向の重複を削除
    # datasetのL_v_paramsは (N_spatial * nt, 2) なので、各レイの先頭だけ取る
    L_v_params_spatial = L_v_params.view(N_spatial, params['nt'], 2)[:, 0, :].to(device)

    # 波数 k の計算
    k_wavenumber = (2 * np.pi * actual_freq / params['c']).item()
    params['k'] = k_wavenumber

    # 5. モデル構築
    # 入力: (x, y) -> 出力: 2ch (Real, Imag)
    model = PINNModel(
        in_features=2,       # 入力次元数
        out_features=2,      # 出力次元数
        hidden_layers=2,     # 中間層の数 (SirenLayerの数。最初の層を除く)
        hidden_features=64,  # 中間層の入出力数
        first_omega_0=10.0,  # 入力層のオメガ
        hidden_omega_0=1.0   # 中間層のオメガ
    ).to(device)
    
    # 6. 学習用データプロバイダ
    def data_provider():
        # コロケーションポイント: 時間tは不要なので (x, y) のみ生成
        # datasetにspatial専用メソッドがなければ自作するか、既存のt=0断面などを使う
        # ここでは簡易的に自作生成
        X_p = torch.rand(params['N_COLLOCATION'], 2).to(device)
        X_p = X_p * params['L'] - params['L']/2 # Centered at 0
        
        # 教師データ: 固定された複素数データ (Real, Imagに分割してもよいし、Loss側で処理してもよい)
        # Loss関数側で複素数として扱うため、そのまま渡す
        return {
            'X_p': X_p,            # PDE用 (x, y)
            'V_v': V_v_complex,    # Data Loss用 (複素数Tensor)
            'L_v_params': L_v_params_spatial # (theta, u)
        }

    # 7. コンポーネント設定
    pde = HelmholtzEquation2D(k=k_wavenumber)
    # dataset作成時と同じ解像度(params['beam_res'])を指定して初期化！
    operator = ProjectionOperator(dataset, n_samples=params['beam_res'])
    loss_fn = PINNLoss_Helmholtz(pde, operator, lambda_param=params['LAMBDA'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'])
    
    # 8. 学習実行
    # train_modelは汎用的に作られている前提
    model, loss_history = train_model(
        model, 
        loss_fn, 
        optimizer, 
        data_provider, 
        params, 
        device
    )
    
    # 9. 結果の保存・表示
    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
        plot_losses(loss_history, save_path=os.path.join(save_dir, "loss.png"), show_plots=False)
        
        # 複素音圧場のプロット
        plot_complex_field(model, dataset, actual_freq.item(), save_dir=save_dir, show_plots=False)
        
        print("保存が完了しました。")
    else:
        plot_losses(loss_history, show_plots=True)
        plot_complex_field(model, dataset, actual_freq.item(), save_dir=None, show_plots=True)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="SingleFreqHelmholtz_frq2")
    parser.add_argument("--preset", type=str, default="default_preset")
    parser.add_argument("--load_preset", action="store_true")
    args = parser.parse_args()
    main(args)