import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import os


# --- 補助関数: 計算ロジックの分離 ---
def _calculate_nmse(p_true, p_est, i_max):
    """正規化平均二乗誤差(NMSE)を計算する"""
    p_range = p_true.max() - p_true.min()
    if p_range.item() < 1e-6: 
        if i_max > 1e-6:
            return torch.sqrt(torch.mean((p_true - p_est)**2)).item() / i_max
        else:
            return torch.sqrt(torch.mean((p_true - p_est)**2)).item()
            
    rmse = torch.sqrt(torch.mean((p_true - p_est)**2))
    return (rmse / p_range).item()

def _calculate_cs(p_true, p_est):
    """コサイン類似度(CS)を計算する"""
    numerator = torch.sum(p_true * p_est)
    denominator = torch.sqrt(torch.sum(p_true**2) * torch.sum(p_est**2))
    if denominator.item() < 1e-9: return 1.0
    return (numerator / denominator).item()

# --- 機能別プロット関数 ---
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
        
    if not show_plots:
        plt.close(fig)

def plot_sinogram(dataset, V_v_full, save_dir=None, show_plots=True):
    """図4: 射影データ（シノグラム）の可視化"""
    N_spatial = dataset.n_angles * dataset.n_rays
    V_v_reshaped = V_v_full.view(N_spatial, dataset.nt).transpose(0, 1).cpu().detach().numpy()
    time_labels = np.linspace(0.0, dataset.T, dataset.nt)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    v_max = 1
    v_min = -v_max
    
    im = ax.imshow(
        V_v_reshaped,
        aspect='auto',
        cmap='seismic',
        interpolation='nearest',
        extent=[0, N_spatial, time_labels.max(), time_labels.min()],
        vmin=v_min, vmax=v_max
    )
    
    ax.set_title(f'Projection Data (Sinogram) - {dataset.n_angles} Angles x {dataset.n_rays} Rays')
    ax.set_xlabel('Measurement Index (Angle x Ray Offset)')
    ax.set_ylabel('Time (t)')
    
    for i in range(1, dataset.n_angles):
        ax.axvline(x=i * dataset.n_rays, color='black', linestyle='--', alpha=0.3)
        
    plt.colorbar(im, ax=ax, label='Line Integral Value (Normalized)')
    
    if save_dir:
        fig.savefig(os.path.join(save_dir, "fig4_synogram.png"))
        
    if not show_plots:
        plt.close(fig)

def compute_and_plot_metrics(model, dataset, sl=100, save_dir=None, show_plots=True):
    """図3: 定量的誤差（NMSE, CS）の計算と時系列プロット"""
    device = next(model.parameters()).device
    time_points_val = np.linspace(0.0, dataset.T, 20) 
    nmse_series = []
    cs_series = []
    
    with torch.no_grad():
        for t_val in time_points_val:
            X_time, P_true_time, _ = dataset.generate_validation_data(sl, sl, 1, frames=[t_val])
            P_est_time = model(X_time.to(device)).cpu().detach()
            P_true_time = P_true_time.cpu()
            
            nmse_series.append(_calculate_nmse(P_true_time, P_est_time, dataset.I_max))
            cs_series.append(_calculate_cs(P_true_time, P_est_time))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # NMSE Plot
    ax[0].plot(time_points_val, nmse_series, label='PINN NMSE', marker='o')
    ax[0].set_title('Normalized Mean Square Error (NMSE) over Time')
    ax[0].set_xlabel('Time (t)')
    ax[0].set_xlim(2*11/26, 2*18/26)
    ax[0].set_ylabel('NMSE')
    ax[0].set_ylim(0, 0.4)
    ax[0].grid(True)
    
    # CS Plot
    ax[1].plot(time_points_val, cs_series, label='PINN CS', marker='o', color='red')
    ax[1].set_title('Cosine Similarity (CS) over Time')
    ax[1].set_xlabel('Time (t)')
    ax[1].set_xlim(2*11/26, 2*18/26)
    ax[1].set_ylabel('CS')
    ax[1].set_ylim(min(np.min(cs_series) - 0.1, 0), 1.05)
    ax[1].grid(True)
    
    fig.suptitle("Quantitative Error Metrics (Fig. 3 Emulation)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_dir:
        fig.savefig(os.path.join(save_dir, "fig3_metrics.png"))
        
    if not show_plots:
        plt.close(fig)

def plot_snapshots(model, dataset, sl=100, save_dir=None, show_plots=True):
    """図2: 2D波面スナップショットのプロット"""
    device = next(model.parameters()).device
    frames = np.linspace(dataset.T * 11/26, dataset.T * 18/26, 4).tolist()
    X_val, P_val_true, frames = dataset.generate_validation_data(sl, sl, len(frames), frames=frames)
    
    with torch.no_grad():
        P_val_est = model(X_val.to(device)).cpu().detach()
    
    P_val_true_reshaped = P_val_true.view(len(frames), sl, sl).numpy()
    P_val_est_reshaped = P_val_est.view(len(frames), sl, sl).numpy()
    
    v_max_plot = 1.0
    v_min_plot = -1.0
    
    fig, axes = plt.subplots(len(frames), 2, figsize=(8, 4 * len(frames)), layout='constrained')
    
    im_est = None 
    
    for i in range(len(frames)):
        # Reference Plot
        ax_true = axes[i, 0]
        im_true = ax_true.imshow(
            P_val_true_reshaped[i], 
            cmap='seismic', vmin=v_min_plot, vmax=v_max_plot, 
            extent=[-dataset.L/2, dataset.L/2, -dataset.L/2, dataset.L/2],
            origin='lower'
        )
        ax_true.set_title(f"Reference (t={frames[i]:.2f})")
        ax_true.set_aspect('equal')
        ax_true.set_xlabel('X'); ax_true.set_ylabel('Y')
        
        # Estimate Plot
        ax_est = axes[i, 1]
        im_est = ax_est.imshow(
            P_val_est_reshaped[i], 
            cmap='seismic', vmin=v_min_plot, vmax=v_max_plot, 
            extent=[-dataset.L/2, dataset.L/2, -dataset.L/2, dataset.L/2],
            origin='lower'
        )
        ax_est.set_title(f"PINN Estimate (t={frames[i]:.2f})")
        ax_est.set_aspect('equal')
        ax_est.set_xlabel('X')
        
        fig.colorbar(im_est, ax=axes[i, :], shrink=0.8, pad=0.02, label='Pressure')

    fig.suptitle("Pressure Field Snapshots (Fig. 2 Emulation)", fontsize=14)
    
    if save_dir:
        fig.savefig(os.path.join(save_dir, "fig2_snapshots.png"))
        
    if not show_plots:
        plt.close(fig)
       
def generate_animation(model, dataset, save_dir=None, output_filename="wave_extrap_animation.gif", show_plots=False):
    print(f"\n--- アニメーション生成 ---")
    device = next(model.parameters()).device
    model.eval()
    
    # --- 設定 ---
    frames, resolution, fps = 60, 150, 15
    view_range = dataset.L
    x = torch.linspace(-view_range, view_range, resolution)
    y = torch.linspace(-view_range, view_range, resolution)
    XX, YY = torch.meshgrid(x, y, indexing='xy')
    t_vals = np.linspace(0, dataset.T, frames)

    # --- プロット準備 ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.zeros((resolution, resolution)), cmap='seismic', vmin=-1, vmax=1,
                   extent=[-view_range, view_range, -view_range, view_range], origin='lower')
    
    # 学習領域の枠線
    L_tr = dataset.L
    rect = patches.Rectangle((-L_tr/2, -L_tr/2), L_tr, L_tr, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(-L_tr/2, L_tr/2 + (view_range * 0.05), "Training Domain", color='black', fontsize=9)
    ax.set_title(f"Extrapolation View: [-{view_range:.1f}, {view_range:.1f}]")
    plt.colorbar(im, ax=ax)

    # --- 更新関数 ---
    def update(frame_idx):
        t = t_vals[frame_idx]
        X_in = torch.stack([XX.flatten(), YY.flatten(), torch.full_like(XX, t).flatten()], dim=1).to(device)
        with torch.no_grad(): 
            P_pred = model(X_in).cpu().view(resolution, resolution).numpy()
        im.set_data(P_pred)
        return im, rect

    # アニメーション作成
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)
    
    # --- 保存処理 ---
    if save_dir:
        save_path = os.path.join(save_dir, output_filename)
        print(f"保存中: {save_path} ...")
        ani.save(save_path, writer='pillow', fps=fps)
        print("保存完了！")
    
    # --- 表示/終了処理 ---
    if show_plots:
        # 表示する場合は、figureを閉じずに ani オブジェクトを返す
        # (返さないとメモリから消えて動かないずら！)
        return ani
    else:
        plt.close(fig)
        return None