import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import glob
import sys
import importlib.util

# このスクリプトがあるフォルダのパス
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_module_from_path(module_name, file_path):
    """
    指定されたファイルパスからモジュールを読み込む関数ずら
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def find_original_script(log_dir):
    """
    ログフォルダの情報から、元の学習に使ったスクリプトを探すずら
    """
    # params.jsonを探す (log_dir内、またはその親)
    params_path = os.path.join(log_dir, "params.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(log_dir, "..", "params.json")
    
    # このスクリプトと同じフォルダにある .py ファイルを探す
    search_pattern = os.path.join(BASE_DIR, "*.py")
    py_files = glob.glob(search_pattern)
    py_filenames = [os.path.basename(f) for f in py_files]

    # SIRENが含まれるかどうかで判定
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
        if candidates:
            target_script_name = candidates[0]
        else:
            target_script_name = "wveq2d_para_mltdeg.py"

    return os.path.join(BASE_DIR, target_script_name)

def create_animation(log_dir, output_filename="wave_animation.gif", script_path=None):
    print(f"フォルダを確認中ずら: {log_dir}")
    
    # --- モジュールの動的ロード ---
    if script_path is None:
        script_path = find_original_script(log_dir)
        
    print(f"使用するスクリプトファイル: {script_path}")
    
    if not os.path.exists(script_path):
        print(f"エラー: スクリプトファイル {script_path} が見つからないずら！")
        return

    try:
        module = load_module_from_path("original_script", script_path)
        PINNModel = module.PINNModel
        AcousticTomographyDataset = module.AcousticTomographyDataset
    except Exception as e:
        print(f"エラー: モジュールの読み込みに失敗したずら: {e}")
        return

    # --- パラメータのロード ---
    params_path = os.path.join(log_dir, "params.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(log_dir, "..", "params.json")

    if not os.path.exists(params_path):
        print("エラー: params.json が見つからないずら...")
        return

    with open(params_path, 'r') as f:
        params = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- データセットとモデルの準備 ---
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
         print("警告: データセットの初期化パラメータが一致しなかったため、デフォルト値を使用したずら。")

    try:
        model = PINNModel(c=dataset.c).to(device)
    except:
        model = PINNModel().to(device)

    # --- 重みのロード ---
    model_path = os.path.join(log_dir, "model.pth")
    if not os.path.exists(model_path):
        print(f"エラー: model.pth が {log_dir} にないずら...")
        return
        
    print("モデルをロード中ずら...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"エラー: モデルの構造が一致しないずら。\n詳細: {e}")
        return

    model.eval()

    # --- アニメーション作成 ---
    frames = 60
    resolution = 100
    fps = 15
    
    t_vals = np.linspace(0, dataset.T, frames)
    x = torch.linspace(-dataset.L/2, dataset.L/2, resolution)
    y = torch.linspace(-dataset.L/2, dataset.L/2, resolution)
    XX, YY = torch.meshgrid(x, y, indexing='xy')

    fig, ax = plt.subplots(figsize=(6, 5))
    v_min, v_max = -1.0, 1.0
    
    im = ax.imshow(
        np.zeros((resolution, resolution)), 
        cmap='seismic', 
        vmin=v_min, vmax=v_max,
        extent=[-dataset.L/2, dataset.L/2, -dataset.L/2, dataset.L/2],
        origin='lower'
    )
    fig.colorbar(im, ax=ax, label='Pressure')
    title = ax.set_title("Time: 0.00")

    print("アニメーション生成中ずら...")

    def update(frame_idx):
        t_current = t_vals[frame_idx]
        TT = torch.full_like(XX, t_current)
        X_input = torch.stack([XX.flatten(), YY.flatten(), TT.flatten()], dim=1).to(device)
        
        with torch.no_grad():
            P_pred = model(X_input).cpu().view(resolution, resolution).numpy()
        
        im.set_data(P_pred)
        title.set_text(f"Time: {t_current:.2f}s")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)
    
    save_path = os.path.join(log_dir, output_filename)
    ani.save(save_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"完了ずら！保存先: {save_path}")

def load_latest_log_dir(base_dir_name="log"):
    # ★ 修正: 「同じフォルダの log」と「親フォルダの log」の両方を探すずら
    search_paths = [
        os.path.join(BASE_DIR, base_dir_name),       # ./log
        os.path.join(BASE_DIR, "..", base_dir_name)  # ../log
    ]

    candidates = []
    print(f"ログフォルダを探索中ずら...")
    
    for search_base in search_paths:
        if os.path.exists(search_base):
            # vX フォルダを再帰的に探す
            for root, dirs, files in os.walk(search_base):
                for d in dirs:
                    if d.startswith('v') and d[1:].isdigit():
                        candidates.append(os.path.join(root, d))

    if not candidates:
        return None
    
    # 更新日時が一番新しいものを返す
    latest_dir = max(candidates, key=os.path.getmtime)
    return latest_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal PINN Wave Animation Viewer")
    parser.add_argument("--dir", type=str, default=None, help="ログフォルダのパス")
    parser.add_argument("--script", type=str, default=None, help="使用するPythonスクリプトのパス")
    args = parser.parse_args()

    target_dir = args.dir
    if target_dir is None:
        target_dir = load_latest_log_dir()
        if target_dir:
            print(f"最新のログフォルダを自動選択したずら: {target_dir}")
        else:
             print("ログフォルダが見つからなかったずら。\n学習スクリプトを実行して `log` フォルダが作成されているか確認してほしいずら。")
    
    if target_dir:
        create_animation(target_dir, script_path=args.script)