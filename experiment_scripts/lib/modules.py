import os
import json
    
class ExperimentManager:
    """
    実験のディレクトリ管理、パラメータの読み書き、バージョニングを行うクラス
    """
    def __init__(self, experiment_name, preset_name, base_log_dir="log"):
        self.experiment_name = experiment_name
        self.preset_name = preset_name
        self.base_log_dir = base_log_dir
        
        # プリセットごとのルートフォルダ (例: log/ParaMltiTanh/default_preset)
        self.preset_dir = os.path.join(self.base_log_dir, self.experiment_name, self.preset_name)
        self.params_path = os.path.join(self.preset_dir, "params.json")
        
        # 現在のパラメータ保持用
        self.params = {}

    def load_params(self, default_params, load_preset=False):
        """
        パラメータをロードして返す。
        load_preset=True の場合はファイルから読み込み、default_params を上書きする。
        """
        self.params = default_params.copy()
        
        # CLI引数の設定を反映
        self.params['experiment'] = self.experiment_name
        self.params['preset'] = self.preset_name

        if load_preset:
            try:
                if os.path.exists(self.params_path):
                    with open(self.params_path, 'r') as f:
                        loaded_params = json.load(f)
                    print(f"パラメータを {self.params_path} から読み込みました。")
                    # デフォルト値をロードした値で更新
                    self.params.update(loaded_params)
                    
                    # ただし実験名とプリセット名は引数のものを強制使用（整合性のため）
                    self.params['experiment'] = self.experiment_name
                    self.params['preset'] = self.preset_name
                else:
                    print(f"警告: {self.params_path} が見つかりません。デフォルトを使用します。")
            except Exception as e:
                print(f"パラメータ読み込みエラー: {e}")
        else:
            print("デフォルトのパラメータを使用します。")
            
        return self.params

    def setup_log_dir(self, save_logs=False):
        """
        ログ保存用のバージョンフォルダ (v1, v2...) を作成してパスを返す。
        保存しない場合は None を返す。
        """
        if not save_logs:
            return None

        # プリセットフォルダ作成
        os.makedirs(self.preset_dir, exist_ok=True)

        # バージョン管理 (v1, v2, ...)
        existing_versions = [d for d in os.listdir(self.preset_dir) 
                             if os.path.isdir(os.path.join(self.preset_dir, d)) and d.startswith('v')]
        
        if not existing_versions:
            new_version = 1
        else:
            # vの後の数字を取り出して最大値+1
            versions = []
            for v in existing_versions:
                try:
                    versions.append(int(v.replace('v', '')))
                except ValueError:
                    continue
            new_version = max(versions) + 1 if versions else 1

        save_dir = os.path.join(self.preset_dir, f"v{new_version}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"ログを {save_dir} に保存します。")

        # 現在のパラメータをプリセットファイルとして保存（次回利用のため）
        self._save_params_to_file()
        
        return save_dir

    def _save_params_to_file(self):
        """現在のパラメータを params.json に保存する内部メソッド"""
        try:
            with open(self.params_path, 'w') as f:
                json.dump(self.params, f, indent=4)
            print(f"現在のパラメータを {self.params_path} に保存しました。")
        except Exception as e:
            print(f"エラー: パラメータファイルの保存に失敗しました: {e}")