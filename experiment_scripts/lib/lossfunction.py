import torch
from torch.autograd import grad

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