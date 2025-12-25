import time
import torch

def train_model(model, loss_fn, optimizer, data_generator, params, device):
    """
    汎用的な学習ループ
    model: PyTorchモデル
    loss_fn: 損失関数 (PINNLoss インスタンスなど)
    optimizer: オプティマイザ
    data_generator: 学習に必要なデータを生成・供給する関数またはオブジェクト
    params: 学習パラメータ (Epoch数など)
    """
    N_EPOCHS = params.get("N_EPOCHS", 10000)
    
    # 履歴保存用
    loss_history = {key: [] for key in ['total', 'L_p', 'L_v']} # 必要に応じてキーを追加
    
    print(f"\n--- Training Start (Epochs={N_EPOCHS}) ---")
    start_time = time.time()
    
    model.train()
    
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()
        
        # データの準備 (data_generatorに任せる)
        # 例: コロケーションポイントのサンプリングなど
        batch_data = data_generator() 
        
        # 損失の計算
        loss, loss_dict = loss_fn(model, batch_data)
        
        loss.backward()
        optimizer.step()
        
        # ログ記録
        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            log_str = f"Epoch {epoch}/{N_EPOCHS} | Time: {elapsed:.1f}s | "
            log_str += " | ".join([f"{k}: {v:.4e}" for k, v in loss_dict.items()])
            print(log_str)
            
            for k, v in loss_dict.items():
                if k in loss_history:
                    loss_history[k].append(v)
            
            start_time = time.time()
            
    print("\n--- Training Finished ---")
    return model, loss_history