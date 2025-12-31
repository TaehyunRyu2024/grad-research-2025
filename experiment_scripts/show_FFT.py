import lib.dataset
from lib.dataset import AcousticParallelBeamTomographyDataset
import torch
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 保存パスの設定 (兄弟フォルダ log/FFT_picture)
# ==========================================
# 現在のスクリプトのディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 1つ上の階層(兄弟)の log フォルダ内の FFT_picture を指定
save_dir = os.path.join(current_dir, '..', 'log', 'FFT_picture')

# フォルダがなければ作成 (再帰的に作成可能)
os.makedirs(save_dir, exist_ok=True)
print(f"保存先ディレクトリ: {os.path.abspath(save_dir)}")

# パラメータ設定 (FFTの分解能を確保するためntは少し多めが望ましい)
nt_sim = 256
freq_center = 2.0

# データセットの初期化
dataset_inst = AcousticParallelBeamTomographyDataset(nt=nt_sim, frequency=freq_center)

print("データ生成中...")
T_v, V_v, L_v_params = dataset_inst.generate_projection_data(beam_res=50)

# データの整形
# V_v: (N_spatial * nt, 1) -> (N_spatial, nt)
# 時間軸(nt)方向に対してFFTをかけるため、形状を変更します
N_spatial = dataset_inst.n_angles * dataset_inst.n_rays
V_reshaped = V_v.view(N_spatial, nt_sim)

# FFTの実行 (実数入力なのでrfftを使用)
# dim=1 (時間軸) に沿ってFFT
print("FFT実行中...")
V_fft = torch.fft.rfft(V_reshaped, dim=1)

# 周波数軸の計算
freqs = torch.fft.rfftfreq(nt_sim, d=(dataset_inst.T / nt_sim))

# パワースペクトルの計算 (振幅の絶対値)
# 全レイ(全角度・全ビーム)での平均をとって「全体的な周波数特性」を見ます
V_fft_abs = torch.abs(V_fft)
V_power_spectrum = torch.mean(V_fft_abs, dim=0) # 全空間平均

# プロット
plt.figure(figsize=(10, 6))
plt.plot(freqs.numpy(), V_power_spectrum.numpy(), 'b-', linewidth=2)
plt.axvline(x=freq_center, color='r', linestyle='--', label=f'Center Freq ({freq_center}Hz)')

# 見やすくするための装飾
plt.title(f'Average Frequency Spectrum of Projection Data\n(Center Freq: {freq_center} Hz, Ricker Wavelet)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (Average)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.xlim(0, freq_center * 3) # 中心周波数の3倍くらいまでを表示範囲とする

# 画像を保存
save_path = os.path.join(save_dir, 'spectrum_analysis_frq2.png')
plt.savefig(save_path)
print(f"スペクトル解析画像を保存しました: {save_path}")
plt.show()