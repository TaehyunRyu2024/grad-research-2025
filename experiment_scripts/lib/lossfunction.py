import torch
from torch.autograd import grad
import torch.nn as nn

# --- 物理方程式（PDE）の定義 ---
class WaveEquation2D:
    def __init__(self, c=1.0):
        self.c = c

    def residual(self, model, X_p):
        """
        物理方程式の残差を計算する。
        X_p: コロケーションポイント (x, y, t)
        """
        X_p.requires_grad_(True)
        p_nn = model(X_p)
        
        # 1階微分
        gradients = grad(p_nn, X_p, grad_outputs=torch.ones_like(p_nn), create_graph=True)[0]
        dp_dx = gradients[:, 0:1]
        dp_dy = gradients[:, 1:2]
        dp_dt = gradients[:, 2:3]

        # 2階微分
        d2p_dx2 = grad(dp_dx, X_p, grad_outputs=torch.ones_like(dp_dx), create_graph=True)[0][:, 0:1]
        d2p_dy2 = grad(dp_dy, X_p, grad_outputs=torch.ones_like(dp_dy), create_graph=True)[0][:, 1:2]
        d2p_dt2 = grad(dp_dt, X_p, grad_outputs=torch.ones_like(dp_dt), create_graph=True)[0][:, 2:3]

        # 波動方程式の残差
        c_squared = self.c**2
        residual = (d2p_dx2 + d2p_dy2) - (1.0 / c_squared) * d2p_dt2
        return residual

# --- 観測演算子（データ項）の定義 ---
class ProjectionOperator:
    def __init__(self, dataset, beam_res=50):
        self.dataset = dataset
        self.beam_res = beam_res

    def forward(self, model, T_v, L_v_params):
        """
        モデル出力に対して積分変換を適用し、観測データと比較可能な形式(V_pred)にする。
        """
        dataset = self.dataset
        beam_res = self.beam_res
        
        N_data = T_v.shape[0]
        
        # パラメータ展開
        theta = L_v_params[:, 0].unsqueeze(1)
        u     = L_v_params[:, 1].unsqueeze(1)
        
        # ジオメトリ計算
        ray_dx = torch.cos(theta)
        ray_dy = torch.sin(theta)
        perp_dx = -torch.sin(theta)
        perp_dy = torch.cos(theta)
        
        center_x = u * perp_dx
        center_y = u * perp_dy
        
        Ps_x = center_x - dataset.radius * ray_dx
        Ps_y = center_y - dataset.radius * ray_dy
        Pd_x = center_x + dataset.radius * ray_dx
        Pd_y = center_y + dataset.radius * ray_dy
        
        Vx = Pd_x - Ps_x
        Vy = Pd_y - Ps_y
        L_beam = torch.sqrt(Vx**2 + Vy**2) 
        
        # 積分パスの構築
        s_points = torch.linspace(0.0, 1.0, beam_res).to(T_v.device).float().unsqueeze(0) 
        
        # バッチ処理用に拡張
        Ps_x_rep = Ps_x.repeat(1, beam_res)
        Ps_y_rep = Ps_y.repeat(1, beam_res)
        Vx_rep   = Vx.repeat(1, beam_res)
        Vy_rep   = Vy.repeat(1, beam_res)
        T_v_rep  = T_v.repeat(1, beam_res)
        s_points_rep = s_points.repeat(N_data, 1)
        
        # 座標計算 P(s) = Ps + s * V
        X_coords = (Ps_x_rep + s_points_rep * Vx_rep).view(-1, 1)
        Y_coords = (Ps_y_rep + s_points_rep * Vy_rep).view(-1, 1)
        T_coords = T_v_rep.view(-1, 1)
        
        # NN入力
        P_input = torch.cat([X_coords, Y_coords, T_coords], dim=1)
        
        # 推論
        P_nn_path = model(P_input)
        P_nn_path_reshaped = P_nn_path.view(N_data, beam_res)
        
        # 台形積分
        ds_step_rep = L_beam / (beam_res - 1)
        
        integral_sum = torch.sum(P_nn_path_reshaped[:, 1:-1], dim=1).unsqueeze(1)
        integral_val = integral_sum + (P_nn_path_reshaped[:, 0].unsqueeze(1) + P_nn_path_reshaped[:, -1].unsqueeze(1)) / 2
        integral_val *= ds_step_rep

        return integral_val / dataset.I_max

# --- 汎用損失関数 ---
class PINNLoss:
    def __init__(self, pde_fn, data_operator=None, lambda_param=1.0):
        self.pde_fn = pde_fn
        self.data_operator = data_operator
        self.lambda_param = lambda_param

    def __call__(self, model, batch_data):
        loss_dict = {}
        total_loss = 0.0

        # 1. 物理損失 (Physics Loss)
        if 'X_p' in batch_data:
            residual = self.pde_fn.residual(model, batch_data['X_p'])
            loss_p = torch.mean(residual**2)
            total_loss += loss_p
            loss_dict['L_p'] = loss_p.item()

        # 2. データ損失 (Data Loss)
        if self.data_operator and 'V_v' in batch_data:
            # 観測データの予測値を計算
            V_pred = self.data_operator.forward(model, batch_data['T_v'], batch_data['L_v_params'])
            
            # ここで V_pred が計算されているはずずら！
            loss_v = torch.mean((batch_data['V_v'] - V_pred)**2)
            total_loss += self.lambda_param * loss_v
            loss_dict['L_v'] = loss_v.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
    
class HelmholtzEquation2D:
    """
    ヘルムホルツ方程式: Δu + k^2 u = 0
    入力 u は (Real, Imag) の2チャネルを持つと仮定
    """
    def __init__(self, k):
        self.k = k

    def compute_residual(self, u, x):
        # x: (N, 2) -> (x, y) requires_grad=True
        
        # u: (N, 2) -> (Real, Imag)
        u_real = u[:, 0:1]
        u_imag = u[:, 1:2]
        
        # 1階微分 (Real)
        grads_real = torch.autograd.grad(u_real, x, grad_outputs=torch.ones_like(u_real), 
                                         create_graph=True, retain_graph=True)[0]
        u_x_real = grads_real[:, 0:1]
        u_y_real = grads_real[:, 1:2]
        
        # 2階微分 (Real)
        u_xx_real = torch.autograd.grad(u_x_real, x, grad_outputs=torch.ones_like(u_x_real), 
                                        create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy_real = torch.autograd.grad(u_y_real, x, grad_outputs=torch.ones_like(u_y_real), 
                                        create_graph=True, retain_graph=True)[0][:, 1:2]
                                        
        # 1階微分 (Imag)
        grads_imag = torch.autograd.grad(u_imag, x, grad_outputs=torch.ones_like(u_imag), 
                                         create_graph=True, retain_graph=True)[0]
        u_x_imag = grads_imag[:, 0:1]
        u_y_imag = grads_imag[:, 1:2]
        
        # 2階微分 (Imag)
        u_xx_imag = torch.autograd.grad(u_x_imag, x, grad_outputs=torch.ones_like(u_x_imag), 
                                        create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy_imag = torch.autograd.grad(u_y_imag, x, grad_outputs=torch.ones_like(u_y_imag), 
                                        create_graph=True, retain_graph=True)[0][:, 1:2]

        # ヘルムホルツ残差: (u_xx + u_yy) + k^2 * u = 0
        # これを実部と虚部それぞれで計算
        res_real = (u_xx_real + u_yy_real) + self.k**2 * u_real
        res_imag = (u_xx_imag + u_yy_imag) + self.k**2 * u_imag
        
        return res_real, res_imag
    

class PINNLoss_Helmholtz:
    """
    ヘルムホルツ方程式用のLoss関数
    Data Loss: 複素数の射影データとのMSE
    PDE Loss: 実部・虚部それぞれのPDE残差のMSE
    """
    def __init__(self, pde, operator, lambda_param=1.0):
        self.pde = pde
        self.operator = operator # ProjectionOperator (線積分)
        self.lambda_param = lambda_param
        self.mse = nn.MSELoss()

    def __call__(self, model, data):
        # 1. PDE Loss (Collocation Points)
        X_p = data['X_p'].requires_grad_(True)
        u_pred = model(X_p) # (N, 2)
        
        res_real, res_imag = self.pde.compute_residual(u_pred, X_p)
        loss_pde = self.mse(res_real, torch.zeros_like(res_real)) + \
                   self.mse(res_imag, torch.zeros_like(res_imag))
        
        # 2. Data Loss (Projection Data)
        # 教師データ V_v は (N_rays, 1) の複素数Tensorと想定
        V_true_complex = data['V_v'] 
        V_true_real = V_true_complex.real
        V_true_imag = V_true_complex.imag
        
        L_v_params = data['L_v_params'] # (theta, u)
        
        # モデルによる射影計算 (Radon変換)
        # operatorは座標を受け取って積分パスを生成し、モデル出力を積分すると仮定
        # 実部と虚部を別々に積分する（線形性）
        
        # operatorの実装によっては、一度 (theta, u) から積分点 (X_integ) を生成する必要があるかもしれません
        # ここでは既存のProjectionOperatorがよしなに計らってくれるラッパーであると想定していますが、
        # もし「座標入力 -> 積分値」を一発で行う関数なら以下のように呼び出します
        
        # 注意: モデル出力が2chなので、Operatorがどう扱うかによります。
        # 単純化のため、モデルのforward関数をラップして渡すテクニックを使います。
        
        def model_real_wrapper(x):
            return model(x)[:, 0:1]
            
        def model_imag_wrapper(x):
            return model(x)[:, 1:2]
            
        # 射影値の予測 (operatorの実装依存ですが、一般的に以下のようなイメージ)
        # ここでは operator.compute_projection(model_func, L_v_params) のような形を想定
        proj_pred_real = self.operator.compute_projection(model_real_wrapper, L_v_params)
        proj_pred_imag = self.operator.compute_projection(model_imag_wrapper, L_v_params)
        
        loss_data = self.mse(proj_pred_real, V_true_real) + \
                    self.mse(proj_pred_imag, V_true_imag)

        # Total Loss
        total_loss = loss_pde + self.lambda_param * loss_data
        
        return total_loss, {
            "loss_pde": loss_pde.item(),
            "loss_data": loss_data.item(),
            "total_loss": total_loss.item()
        }
    

class ProjectionOperator:
    def __init__(self, dataset, n_samples=50):
        self.dataset = dataset
        self.n_samples = n_samples # dataset.pyの beam_res に相当
        
        # 0.0 ~ 1.0 の等間隔な点 (dataset.pyの s_points と同じ)
        self.s_vals = torch.linspace(0.0, 1.0, steps=self.n_samples).view(1, -1)

    def compute_projection(self, model_func, params):
        """
        dataset.py の generate_projection_data と完全に同じ「台形積分」を行います。
        """
        device = params.device
        N_rays = params.shape[0]
        
        # --- 1. ジオメトリ計算 (ここは前回と同じでOK) ---
        theta = params[:, 0:1]
        u = params[:, 1:2]

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # Datasetと同じ計算ロジック
        # vec_u (perp) = (-sin, cos)
        center_x = u * (-sin_t)
        center_y = u * (cos_t)
        
        R = self.dataset.radius
        
        # 始点 Ps, 終点 Pd
        Ps_x = center_x - R * cos_t
        Ps_y = center_y - R * sin_t
        
        Pd_x = center_x + R * cos_t
        Pd_y = center_y + R * sin_t
        
        # レイの長さ L_beam
        # (dataset.pyでは Vx**2 + Vy**2 のルートで計算している値)
        ray_len = torch.sqrt((Pd_x - Ps_x)**2 + (Pd_y - Ps_y)**2)
        
        # --- 2. 座標生成 ---
        s = self.s_vals.to(device).expand(N_rays, self.n_samples)
        
        X_interp = Ps_x + s * (Pd_x - Ps_x)
        Y_interp = Ps_y + s * (Pd_y - Ps_y)
        
        inputs = torch.stack([X_interp.flatten(), Y_interp.flatten()], dim=1)
        
        # --- 3. モデル予測 ---
        # values: (N_rays * n_samples, 1)
        values_flat = model_func(inputs)
        values = values_flat.view(N_rays, self.n_samples)
        
        # --- 4. 台形積分 (Trapezoidal Rule) ---
        # dataset.pyの実装:
        # integral = sum(middle) + (first + last)/2
        # integral *= ds_step
        
        # ステップ幅 ds = L / (N - 1)
        ds_step = ray_len / (self.n_samples - 1)
        
        # 中身の和 (インデックス 1 から -2 まで)
        sum_middle = torch.sum(values[:, 1:-1], dim=1, keepdim=True)
        
        # 端点の和の半分
        sum_ends = (values[:, 0:1] + values[:, -1:]) / 2.0
        
        # 合計してステップ幅をかける
        integral = (sum_middle + sum_ends) * ds_step
        
        return integral