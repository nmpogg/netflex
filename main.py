# Cập nhật đoạn code khởi tạo mô hình trong file main.py của bạn:

from data_loader import load_and_preprocess # Import hàm nâng cao mới
from evaluator import RecSysEvaluator
from model.dqn import AdvancedDQNAgent, ReplayBuffer # Import Agent mới

def run_benchmark():
    log_file = 'data/log_random_4_22_to_5_08_pure.csv'
    
    # Gọi hàm advanced thu về tập dữ liệu cấu trúc mới
    train_data, test_data, all_videos = load_and_preprocess(log_file, seq_len=3)
    
    num_videos = max(all_videos) + 1
    
    # Khai báo Agent với các tham số chiều dữ liệu nâng cao
    models = {
        "DQN_Advanced": AdvancedDQNAgent(
            num_videos=num_videos, 
            video_embed_dim=64, 
            behavior_dim=4, # 4 cột tín hiệu: watch_ratio, like, comment, hate
            seq_len=3
        ),
        # "Proposed_Model": Mô hình đề xuất cải tiến của bạn sẽ cắm vào đây
    }
    
    # Các bước chạy vòng lặp huấn luyện phía dưới giữ nguyên hoàn toàn...
    
    # 3. Tiến hành Benchmark từng mô hình
    for model_name, agent in models.items():
        print(f"\n{'='*50}\nBẮT ĐẦU HUẤN LUYỆN: {model_name}\n{'='*50}")
        
        replay_buffer = ReplayBuffer(capacity=50000)
        epochs = 50
        batch_size = 256
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Giai đoạn Online/Offline Train
            for index, row in train_data.iterrows():
                # Đẩy dữ liệu vào Replay Buffer
                replay_buffer.push(
                    row['state_history'], 
                    row['action_video'], 
                    row['reward'], 
                    row['next_state_history'], 
                    row['done']
                )
                
                # Bắt đầu học khi buffer đủ lớn
                loss = agent.train_step(replay_buffer, batch_size=batch_size)
                total_loss += loss
            
            # Giai đoạn Evaluation (Kiểm định)
            print(f"Đang đánh giá {model_name} ở Epoch {epoch+1}...")
            hr, ndcg, mmap = RecSysEvaluator.evaluate_epoch(agent, test_data, all_videos, k=10)
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_data):.4f}")
            print(f"Metrics (Top 10): Hit Rate = {hr:.4f} | NDCG = {ndcg:.4f} | MAP = {mmap:.4f}")

if __name__ == "__main__":
    run_benchmark()