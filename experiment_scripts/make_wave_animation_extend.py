import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import argparse
import glob
import sys
import importlib.util

# このスクリプトがあるフォルダのパス
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_module_from_path(module_name, file_path):
    """指定されたファイルパスからモジュールを読み込むずら"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None: return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def find_original_script(log_dir):
    """ログフォルダの情報から、元の学習に使ったスクリプトを探すずら"""
    params_path = os.path.join(log_dir, "params.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(log_dir, "..", "params.json")
    
    search_pattern = os.path.join(BASE_DIR, "*.py")
    py_files = glob.glob(search_pattern)
    py_filenames = [os.path.basename(f) for f in py_files]

    is_siren = "SIREN" in log_dir or "siren" in log_dir.lower()
    target_script_name = None

    if is_siren:
        for fname in py_filenames:
            if "SIREN" in fname and "make_animation" not in fname:
                target_script_name = fname
                break
    
    if target_script_name is None:
         for fname in py_filenames:
            if "wveq2d_para_mltdeg.py" in fname:
                target_script_name = fname
                break
    
    if target_script_name is None:
        candidates = [f for f in py_filenames if "make_animation" not in f]
        if candidates: target_script_name = candidates[0]
        else: target_script_name = "wveq2d_para_mltdeg.py"

    return os.path.join(BASE_DIR, target_script_name)

def create_extrapolation_animation(log_dir, output_filename="wave_extrap_animation_extend.gif", script_path=None):
    print(f"フォルダを確認中ずら: {log_dir}")
    
    # --- モジュールのロード ---
    if script_path is None:
        script_path = find_original_script(log_dir)
        
    print(f"使用するスクリプト: {script_path}")
    if not os.path.exists(script_path):
        print(f"エラー: スクリプト {script_path} が見つからないずら！")
        return

    try:
        module = load_module_from_path("original_script", script_path)
        PINNModel = module.PINNModel
        AcousticTomographyDataset = module.AcousticTomographyDataset
    except Exception as e:
        print(f"エラー: モジュールの読み込み失敗ずら: {e}")
        return

    # --- パラメータロード ---
    params_path = os.path.join(log_dir, "params.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(log_dir, "..", "params.json")

    if not os.path.exists(params_path):
        print("エラー: params.json がないずら...")
        return

    with open(params_path, 'r') as f:
        params = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- データセット準備 ---
    try:
        dataset = AcousticTomographyDataset(
            L=params.get('L', 1.0), 
            T=params.get('T', 2.0), 
            c=params.get('c', 1.0), 
            nl=params.get('nl', 20), 
            n_angles=params.get('n_angles', 18),
            nt=params.get('nt', 26)
        )
    except TypeError:
         dataset = AcousticTomographyDataset()
    
    # 学習時の領域サイズ (L)
    # 本来の学習範囲は [-L/2, L/2]
    L_training = dataset.L

    # --- モデル準備 ---
    try:
        model = PINNModel(c=dataset.c).to(device)
    except:
        model = PINNModel().to(device)

    model_path = os.path.join(log_dir, "model.pth")
    if not os.path.exists(model_path):
        print(f"エラー: model.pth がないずら...")
        return
        
    print("モデルをロード中ずら...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"エラー: モデル構造不一致ずら。\n詳細: {e}")
        return

    model.eval()

    # --- アニメーション設定 (外挿用) ---
    frames = 60
    resolution = 150
    fps = 15
    
    # ★ ここが変更点: 表示範囲を [-L, L] に設定するずら
    # dataset.L をそのまま使うことで、L=1.0なら[-1, 1], L=2.0なら[-2, 2]になるずら
    view_range = dataset.L
    
    x = torch.linspace(-view_range, view_range, resolution)
    y = torch.linspace(-view_range, view_range, resolution)
    XX, YY = torch.meshgrid(x, y, indexing='xy')
    
    t_vals = np.linspace(0, dataset.T, frames)

    fig, ax = plt.subplots(figsize=(6, 5))
    v_min, v_max = -1.0, 1.0
    
    # 外挿範囲を描画
    im = ax.imshow(
        np.zeros((resolution, resolution)), 
        cmap='seismic', 
        vmin=v_min, vmax=v_max,
        extent=[-view_range, view_range, -view_range, view_range],
        origin='lower'
    )
    
    # ★ 学習領域 (ROI) の枠線を表示するずら
    # [-L/2, L/2] の範囲を四角で囲む
    rect = patches.Rectangle(
        (-L_training/2, -L_training/2), # 左下座標
        L_training, L_training,         # 幅, 高さ
        linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)
    
    # 説明用のテキストを追加
    ax.text(-L_training/2, L_training/2 + (view_range * 0.05), "Training Domain", color='black', fontsize=9, ha='left')

    fig.colorbar(im, ax=ax, label='Pressure')
    title = ax.set_title(f"Extrapolation View: [-{view_range:.1f}, {view_range:.1f}]")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    print(f"外挿アニメーション生成中 (Range: [-{view_range}, {view_range}])...")

    def update(frame_idx):
        t_current = t_vals[frame_idx]
        TT = torch.full_like(XX, t_current)
        X_input = torch.stack([XX.flatten(), YY.flatten(), TT.flatten()], dim=1).to(device)
        
        with torch.no_grad():
            P_pred = model(X_input).cpu().view(resolution, resolution).numpy()
        
        im.set_data(P_pred)
        title.set_text(f"Time: {t_current:.2f}s")
        return im, title, rect

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)
    
    save_path = os.path.join(log_dir, output_filename)
    ani.save(save_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"完了ずら！保存先: {save_path}")

def load_latest_log_dir(base_dir_name="log"):
    search_paths = [
        os.path.join(BASE_DIR, base_dir_name),
        os.path.join(BASE_DIR, "..", base_dir_name)
    ]
    candidates = []
    for search_base in search_paths:
        if os.path.exists(search_base):
            for root, dirs, files in os.walk(search_base):
                for d in dirs:
                    if d.startswith('v') and d[1:].isdigit():
                        candidates.append(os.path.join(root, d))
    if not candidates: return None
    return max(candidates, key=os.path.getmtime)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN Extrapolation Animation Viewer")
    parser.add_argument("--dir", type=str, default=None, help="ログフォルダパス")
    parser.add_argument("--script", type=str, default=None, help="スクリプトパス")
    args = parser.parse_args()

    target_dir = args.dir
    if target_dir is None:
        target_dir = load_latest_log_dir()
        if target_dir:
            print(f"最新ログフォルダ自動選択: {target_dir}")
        else:
             print("ログフォルダが見つからなかったずら。")
    
    if target_dir:
        create_extrapolation_animation(target_dir, script_path=args.script)