from data_loader import load_video_features, load_logs_and_build_mdp
from evaluator import RecSysEvaluator
from models.hybrid_dqn import HybridDQNAgent, ReplayBuffer

def run():
    # 1. Đường dẫn file (Sửa lại cho đúng với thư mục của bạn)
    feature_file = 'data/video_features_basic_pure.csv'
    log_file = 'data/log_random_4_22_to_5_08_pure.csv'
    
    # 2. Pipeline Dữ liệu
    feat_dict, num_videos, num_authors, num_musics = load_video_features(feature_file)
    train_data, test_data, all_vids = load_logs_and_build_mdp(log_file)
    
    # --- DEBUG NHANH (Bỏ comment 2 dòng dưới để chạy nháp 1 phút trước khi chạy thật) ---
    # train_data = train_data.head(500)
    # test_data = test_data.head(100)
    
    # 3. Khởi tạo Mô hình
    agent = HybridDQNAgent(feat_dict, num_videos, num_authors, num_musics)
    replay_buffer = ReplayBuffer(capacity=50000)
    
    # 4. Huấn luyện
    epochs = 5
    batch_size = 256
    
    for epoch in range(epochs):
        print(f"\n[{'-'*15} EPOCH {epoch+1}/{epochs} {'-'*15}]")
        total_loss = 0
        
        for index, row in train_data.iterrows():
            replay_buffer.push(row['state_history'], row['action_video'], 
                               row['reward'], row['next_state_history'], row['done'])
            
            loss = agent.train_step(replay_buffer, batch_size)
            total_loss += loss
            
        avg_loss = total_loss / len(train_data)
        print(f"Loss trung bình: {avg_loss:.4f}")
        
        print("Đang đánh giá trên tập Test...")
        hr, ndcg, mmap = RecSysEvaluator.evaluate_epoch(agent, test_data, all_vids)
        print(f"Metrics Top-10: HR = {hr:.4f} | NDCG = {ndcg:.4f} | MAP = {mmap:.4f}")

if __name__ == "__main__":
    run()