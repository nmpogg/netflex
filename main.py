import os
import torch
from data_loader import load_video_features, load_user_features, load_logs_and_build_mdp
from evaluator import RecSysEvaluator
from models.hybrid_dqn import HybridDQNAgent, ReplayBuffer

def run_benchmark():
    # 1. Đường dẫn thư mục Data
    feature_file = 'data/video_features_basic_pure.csv'
    user_file = 'data/user_features_pure.csv'
    log_file = 'data/log_random_4_22_to_5_08_pure.csv'
    
    # 2. Xử lý Dữ liệu
    video_feat_dict, num_videos, num_authors, num_musics = load_video_features(feature_file)
    user_feat_dict, num_users, user_dim = load_user_features(user_file)
    train_data, test_data, all_vids = load_logs_and_build_mdp(log_file)
    
    # ========================================================
    # SANITY CHECK: Bỏ comment 2 dòng này để test thử trước khi chạy toàn bộ file
    # train_data = train_data.head(500)
    # test_data = test_data.head(100)
    # ========================================================
    
    # 3. Khởi tạo DQN Baseline
    agent = HybridDQNAgent(
        video_feat_dict=video_feat_dict,
        user_feat_dict=user_feat_dict,
        num_videos=num_videos,
        num_authors=num_authors,
        num_musics=num_musics,
        num_users=num_users,
        user_dim=user_dim
    )
    
    replay_buffer = ReplayBuffer(capacity=50000)
    epochs = 50 # Giờ bạn có thể tự tin đặt số epoch thật cao (VD: 50 hoặc 100)
    batch_size = 256
    
    # === CẤU HÌNH EARLY STOPPING ===
    best_ndcg = 0.0
    patience = 5        # Số epoch cho phép mô hình "dậm chân tại chỗ"
    patience_counter = 0
    model_save_path = "best_hybrid_dqn.pth"
    # ===============================

    for epoch in range(epochs):
        print(f"\n[{'-'*15} EPOCH {epoch+1}/{epochs} {'-'*15}]")
        total_loss = 0
        
        # 1. Giai đoạn Huấn luyện (Training)
        for index, row in train_data.iterrows():
            replay_buffer.push(
                row['user_id'], row['state_history'], row['action_video'], 
                row['reward'], row['next_state_history'], row['done']
            )
            loss = agent.train_step(replay_buffer, batch_size)
            total_loss += loss
            
        avg_loss = total_loss / len(train_data)
        print(f"Loss trung bình: {avg_loss:.4f}")
        
        # 2. Giai đoạn Đánh giá (Evaluation)
        print("Đang đánh giá trên tập Test...")
        # LƯU Ý: Dùng hàm evaluate_epoch trả về Dictionary nếu bạn đã cập nhật bản Học thuật
        metrics = RecSysEvaluator.evaluate_epoch(agent, test_data, train_data, all_vids)
        current_ndcg = metrics['NDCG']
        
        print(f"HR@10: {metrics['HR']:.4f} | NDCG@10: {current_ndcg:.4f} | Coverage: {metrics['Coverage']*100:.2f}%")
        
        # === LOGIC EARLY STOPPING & SAVE MODEL ===
        if current_ndcg > best_ndcg:
            print(f"🌟 NDCG tăng từ {best_ndcg:.4f} lên {current_ndcg:.4f}. Đang lưu mô hình...")
            best_ndcg = current_ndcg
            patience_counter = 0 # Reset bộ đếm
            
            # Lưu trọng số của Mạng Q-Network chính
            torch.save(agent.q_net.state_dict(), model_save_path)
        else:
            patience_counter += 1
            print(f"⚠️ NDCG không tăng. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n🛑 KÍCH HOẠT EARLY STOPPING TẠI EPOCH {epoch+1}!")
                print(f"Mô hình đã bắt đầu Overfitting. Giữ lại trọng số tốt nhất: NDCG = {best_ndcg:.4f}")
                break # Cắt đứt vòng lặp huấn luyện

    print(f"\n✅ Hoàn tất! Mô hình xuất sắc nhất đã được lưu tại: {os.path.abspath(model_save_path)}")

if __name__ == "__main__":
    run_benchmark()