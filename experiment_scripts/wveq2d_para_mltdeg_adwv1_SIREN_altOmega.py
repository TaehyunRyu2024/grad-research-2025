import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from lib.modules import (
    ExperimentManager
)
from lib.dataset import (
    AcousticParallelBeamTomographyDataset_AddWave1 as APBTDataset
)
from lib.PINN_model import (
    PINNModel_SIREN_altOmega as PINNModel
)
from lib.lossfunction import (
    WaveEquation2D,
    ProjectionOperator,
    PINNLoss
)
from lib.train import (
    train_model
)
from lib.evaluation import (
    plot_sinogram,
    compute_and_plot_metrics,
    plot_snapshots,
    generate_animation,
    generate_animation_x2, 
    plot_losses
)

# ★ 警告を非表示にする設定
import warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")

# PyTorchのシード設定
torch.manual_seed(42)
np.random.seed(42)

# ================================================================
# メイン実行ブロック (実験管理)
# ================================================================

def main(args):
    # --- デフォルトパラメータ ---
    default_params = {
        "N_EPOCHS": 50000,
        "N_COLLOCATION": 100,
        "N_IC": 1000,
        "LR": 1e-3,
        "LAMBDA": 1e4,
        "L": 1.0,
        "T": 2.0,
        "c": 1.2,
        "nl": 20,
        "nt": 26,
        "frequency": 5.0,
        "snr_db": 20.0,
        "beam_res": 50,
        "n_angles": 18 
    }
    
    # 1. マネージャーの初期化
    exp_manager = ExperimentManager(args.experiment, args.preset)
    
    # 2. パラメータのロード (デフォルト + ファイル読み込み)
    params = exp_manager.load_params(default_params, load_preset=args.load_preset)

    # 3. ログ保存の確認とディレクトリ作成
    save_logs_input = input("この実行のログ（モデル、プロット）を保存しますか？ (y/n): ").strip().lower()
    save_dir = exp_manager.setup_log_dir(save_logs=(save_logs_input == 'y'))
    
    # 保存しない場合はプロットを表示する
    show_plots = (save_dir is None)

    # 実行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # データセット
    dataset = APBTDataset(
        L=params['L'], T=params['T'], c=params['c'], 
        nl=params['nl'], n_angles=params.get('n_angles', 18),
        nt=params['nt'], frequency=params['frequency'], snr_db=params['snr_db']
    )
    
    # データ生成
    print("射影データ生成中...")
    T_v, V_v, L_v_params = dataset.generate_projection_data(beam_res=params['beam_res'])
    T_v, V_v, L_v_params = T_v.to(device), V_v.to(device), L_v_params.to(device)
    
    # モデル
    model = PINNModel(c=dataset.c).to(device)
    
    # 学習
    # データ生成器の定義（実験ごとにここを変える）
    def data_provider():
        # 毎回ランダムにコロケーションポイントを生成する場合
        X_p = dataset.generate_pinn_collocation_points(params['N_COLLOCATION']).to(device)
    
        # 固定データ（観測データなど）
        return {
            'X_p': X_p,
            'T_v': T_v,
            'V_v': V_v,
            'L_v_params': L_v_params
        }

    # コンポーネントの組み立て
    pde = WaveEquation2D(c=params['c'])
    operator = ProjectionOperator(dataset) # データセット固有の変換ロジック
    loss_fn = PINNLoss(pde, operator, lambda_param=params['LAMBDA'])
    # Adamを使いたい場合
    optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'])
    
    # 学習実行
    model, loss_history = train_model(
        model, 
        loss_fn, 
        optimizer, 
        data_provider, 
        params, 
        device
    )
    # 結果処理
    if save_dir:
        # 保存モード
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
        plot_losses(loss_history, save_path=os.path.join(save_dir, "loss.png"), show_plots=False)# 個別関数呼び出し
        plot_sinogram(dataset, V_v, save_dir=save_dir, show_plots=False)
        compute_and_plot_metrics(model, dataset, save_dir=save_dir, show_plots=False)
        plot_snapshots(model, dataset, save_dir=save_dir, show_plots=False)
        generate_animation(model, dataset, save_dir=save_dir, show_plots=False)
        generate_animation_x2(model, dataset, save_dir=save_dir, show_plots=False)
        
        dataset.plot_geometry(save_path=os.path.join(save_dir, "geometry.png"))
        print("保存が完了しました。")
        
    else:
        # 表示モード (保存しない)
        plot_losses(loss_history, show_plots=True)
        plot_sinogram(dataset, V_v, save_dir=None, show_plots=True)
        compute_and_plot_metrics(model, dataset, save_dir=None, show_plots=True)
        plot_snapshots(model, dataset, save_dir=None, show_plots=True)
        
        # ★重要: 変数 'ani' に入れないと動かないずら！
        # save_dir=None, show_plots=True で呼び出す
        ani = generate_animation(model, dataset, save_dir=None, show_plots=True)
        
        plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="ParaMltiAddWave1SIREN_altOmega")
    parser.add_argument("--preset", type=str, default="default_preset")
    parser.add_argument("--load_preset", action="store_true")
    args = parser.parse_args()
    main(args)