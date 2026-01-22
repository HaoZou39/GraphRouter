import os
import json
import datetime
import pandas as pd
import numpy as np

class TrainingLogger:
    """Training logger for saving loss, accuracy and other metrics during training"""
    
    def __init__(self, log_dir='../data/training_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filenames with timestamps
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.loss_log_file = os.path.join(log_dir, f'training_loss_{timestamp}.json')
        self.accuracy_log_file = os.path.join(log_dir, f'training_accuracy_{timestamp}.json')
        self.eval_log_file = os.path.join(log_dir, f'evaluation_metrics_{timestamp}.json')
        # 修改：使用固定的loss_components文件名
        self.loss_components_file = os.path.join(log_dir, f'loss_components_{timestamp}.json')
        # 🔧 新增：reward统计日志文件
        self.reward_stats_file = os.path.join(log_dir, f'reward_stats_{timestamp}.json')
        
        # Initialize log data
        self.loss_data = []
        self.accuracy_data = []
        self.eval_data = []
        self.loss_components_log = []  # Add Loss component logs
        self.reward_stats_log = []  # 🔧 新增：reward统计日志
        
        print(f"Training logs will be saved to: {log_dir}")
        print(f"Loss log: {self.loss_log_file}")
        print(f"Accuracy log: {self.accuracy_log_file}")
        print(f"Evaluation log: {self.eval_log_file}")
        print(f"Loss components log: {self.loss_components_file}")
        print(f"Reward stats log: {self.reward_stats_file}")  # 🔧 新增
    
    def log_loss(self, step, loss, learning_rate, phase="phase1"):
        """Log training loss"""
        log_entry = {
            'step': step,
            'loss': float(loss),
            'learning_rate': float(learning_rate),
            'phase': phase,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.loss_data.append(log_entry)
        
        # 性能优化：减少文件写入频率，从100改为500
        if len(self.loss_data) % 500 == 0:
            self.save_loss_log()
    
    def log_accuracy(self, step, accuracy, correct_predictions, total_predictions, phase="phase1", reward=None):
        """Log training accuracy and reward"""
        log_entry = {
            'step': step,
            'accuracy': float(accuracy),
            'correct_predictions': int(correct_predictions),
            'total_predictions': int(total_predictions),
            'phase': phase,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add reward if provided
        if reward is not None:
            log_entry['reward'] = float(reward)
        
        self.accuracy_data.append(log_entry)
        
        # 性能优化：减少文件写入频率，从100改为500
        if len(self.accuracy_data) % 500 == 0:
            self.save_accuracy_log()
    
    def log_evaluation(self, step, metrics):
        """Log evaluation metrics"""
        log_entry = {
            'step': step,
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.eval_data.append(log_entry)
        self.save_eval_log()
    
    def log_loss_components(self, step, loss_components, learning_rate, phase):
        """Log Loss components"""
        # Helper function to safely convert tensor to scalar
        def tensor_to_scalar(tensor):
            if hasattr(tensor, 'numpy'):
                # TensorFlow tensor
                numpy_array = tensor.numpy()
                if numpy_array.size == 1:
                    return float(numpy_array.item())
                else:
                    # Multi-dimensional tensor, take the mean
                    return float(np.mean(numpy_array))
            elif hasattr(tensor, 'item'):
                # NumPy array
                if tensor.size == 1:
                    return float(tensor.item())
                else:
                    # Multi-dimensional array, take the mean
                    return float(np.mean(tensor))
            else:
                # Already a scalar
                return float(tensor)
        
        log_entry = {
            "step": step,
            "total_loss": tensor_to_scalar(loss_components['total_loss']),
            "balanced_loss": tensor_to_scalar(loss_components['balanced_loss']),
            # 两阶段训练损失
            "phase1_sl_only_loss": tensor_to_scalar(loss_components['phase1_sl_only_loss']),
            "phase2_sl_rl_loss": tensor_to_scalar(loss_components['phase2_sl_rl_loss']),
            # 基础损失
            "ce_loss": tensor_to_scalar(loss_components['ce_loss']),
            "q_loss": tensor_to_scalar(loss_components['q_loss']),
            "q_loss_base": tensor_to_scalar(loss_components['q_loss_base']),
            "q_loss_weighted": tensor_to_scalar(loss_components['q_loss_weighted']),
            # Q学习损失组件
            "qloss_positive": tensor_to_scalar(loss_components['qloss_positive']),

            # 训练状态
            "training_phase": tensor_to_scalar(loss_components['training_phase']),
            "rl_weight": tensor_to_scalar(loss_components['rl_weight']),

            # Mask和Q值统计
            "mask_penalty": tensor_to_scalar(loss_components['mask_penalty']),
            "q_value_range": tensor_to_scalar(loss_components['q_value_range']),
            "q_value_min": tensor_to_scalar(loss_components['q_value_min']),
            "q_value_max": tensor_to_scalar(loss_components['q_value_max']),
            "q_value_mean": tensor_to_scalar(loss_components['q_value_mean']),
            "q_value_std": tensor_to_scalar(loss_components['q_value_std']),


            # 未mask的Q值统计
            "q_value_unmasked_max": tensor_to_scalar(loss_components['q_value_unmasked_max']),
            "q_value_unmasked_min": tensor_to_scalar(loss_components['q_value_unmasked_min']),
            "mask_penalty_gap": tensor_to_scalar(loss_components['mask_penalty_gap']),
            # 训练元信息
            "learning_rate": float(learning_rate),
            "phase": phase,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.loss_components_log.append(log_entry)
        
        # Save every 1000 steps
        if step % 1000 == 0:
            self._save_loss_components()

    def log_reward_stats(self, step, reward_data):
        """🔧 新增：记录reward统计信息"""
        log_entry = {
            'step': step,
            'phase': reward_data['phase'],
            'rl_weight': reward_data['rl_weight'],
            'all_history_reward_sum': reward_data['all_history_reward_sum'],
            'all_history_reward_mean': reward_data['all_history_reward_mean'],
            'all_history_reward_std': reward_data['all_history_reward_std'],
            'all_history_reward_min': reward_data['all_history_reward_min'],
            'all_history_reward_max': reward_data['all_history_reward_max'],
            'all_history_reward_count': reward_data['all_history_reward_count'],
            'positive_reward_count': reward_data['positive_reward_count'],
            'positive_reward_mean': reward_data['positive_reward_mean'],
            'positive_reward_min': reward_data['positive_reward_min'],
            'positive_reward_max': reward_data['positive_reward_max'],
            'negative_reward_count': reward_data['negative_reward_count'],
            'negative_reward_mean': reward_data['negative_reward_mean'],
            'negative_reward_min': reward_data['negative_reward_min'],
            'negative_reward_max': reward_data['negative_reward_max'],
            'reward_range': reward_data['reward_range'],
            'positive_negative_ratio': reward_data['positive_negative_ratio'],
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.reward_stats_log.append(log_entry)
        
        # Save every 1000 steps
        if step % 1000 == 0:
            self._save_reward_stats()

    def _save_loss_components(self):
        """Save Loss component logs to fixed file"""
        if self.loss_components_log:
            # 修改：使用固定的文件名，覆盖写入
            with open(self.loss_components_file, 'w') as f:
                json.dump(self.loss_components_log, f, indent=2)
            print(f"Loss components saved to: {self.loss_components_file}")
    
    def _save_reward_stats(self):
        """🔧 新增：保存reward统计日志"""
        if self.reward_stats_log:
            with open(self.reward_stats_file, 'w') as f:
                json.dump(self.reward_stats_log, f, indent=2)
            print(f"Reward stats saved to: {self.reward_stats_file}")

    def save_loss_log(self):
        """Save loss log to JSON file"""
        with open(self.loss_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.loss_data, f, indent=2, ensure_ascii=False)
    
    def save_accuracy_log(self):
        """Save accuracy log to JSON file"""
        with open(self.accuracy_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.accuracy_data, f, indent=2, ensure_ascii=False)
    
    def save_eval_log(self):
        """Save evaluation log to JSON file"""
        with open(self.eval_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.eval_data, f, indent=2, ensure_ascii=False)
    
    def save_all_logs(self):
        """Save all logs"""
        self.save_loss_log()
        self.save_accuracy_log()
        self.save_eval_log()
        self._save_loss_components()  # 确保loss components也被保存
        self._save_reward_stats()  # 🔧 新增：确保reward stats也被保存
        print("All training logs saved successfully!")
    
    def get_latest_metrics(self):
        """Get latest training metrics"""
        latest_metrics = {}
        
        if self.loss_data:
            latest_metrics['latest_loss'] = self.loss_data[-1]
        
        if self.accuracy_data:
            latest_metrics['latest_accuracy'] = self.accuracy_data[-1]
        
        if self.eval_data:
            latest_metrics['latest_eval'] = self.eval_data[-1]
        
        return latest_metrics
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.loss_data or not self.accuracy_data:
                print("Not enough data available for plotting")
                return
            
            df = pd.DataFrame({
                'step': [entry['step'] for entry in self.loss_data],
                'loss': [entry['loss'] for entry in self.loss_data],
                'accuracy': [entry['accuracy'] for entry in self.accuracy_data]
            })
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            
            # Loss curve
            ax.plot(df['step'], df['loss'], label='Training Loss')
            ax.set_title('Training Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.grid(True)
            
            # Accuracy curve
            ax.plot(df['step'], df['accuracy'], label='Training Accuracy')
            ax.set_title('Training Accuracy')
            ax.set_xlabel('Step')
            ax.set_ylabel('Accuracy')
            ax.grid(True)
            
            ax.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Training curves saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"Error plotting training curves: {e}")


# Create global logger instance
training_logger = None

def init_logger(log_dir='../data/training_logs'):
    """Initialize global logger"""
    global training_logger
    training_logger = TrainingLogger(log_dir)
    return training_logger

def get_logger():
    """Get global logger"""
    global training_logger
    if training_logger is None:
        training_logger = TrainingLogger()
    return training_logger 