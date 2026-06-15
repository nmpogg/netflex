import torch
import os
from tqdm import tqdm # Thêm thư viện tqdm

from data_loader import load_video_features, load_user_features, load_logs_and_build_mdp
from evaluator import RecSysEvaluator
from models.hybrid_dqn import HybridDQNAgent, ReplayBuffer

def run_benchmark():
    feature_file = 'data/video_features_basic_pure.csv'
    user_file = 'data/user_features_pure.csv'
    log_file = 'data/log_random_4_22_to_5_08_pure.csv'
    
    video_feat_dict, num_videos, num_authors, num_musics = load_video_features(feature_file)
    user_feat_dict, num_users, user_dim = load_user_features(user_file)
    train_data, test_data, all_vids = load_logs_and_build_mdp(log_file)
    
    agent = HybridDQNAgent(video_feat_dict, user_feat_dict, num_videos, num_authors, num_musics, num_users, user_dim)
    replay_buffer = ReplayBuffer(capacity=50000)
    
    epochs = 50 
    batch_size = 1024
    
    best_ndcg = 0.0
    patience = 5        
    patience_counter = 0
    model_save_path = "best_hybrid_dqn.pth"

    print("\n🚀 Đang nạp toàn bộ dữ liệu vào Replay Buffer...")
    for row in tqdm(train_data.itertuples(), total=len(train_data), desc="Nạp Buffer"):
        replay_buffer.push(
            row.user_id, row.state_history, row.action_video, 
            row.reward, row.next_state_history, row.done
        )
        
    num_batches = len(replay_buffer) // batch_size
    print(f"Hoàn tất nạp Buffer. Tổng số Batch mỗi Epoch: {num_batches}")
    # =====================================================================

    for epoch in range(epochs):
        print(f"\n[{'-'*15} EPOCH {epoch+1}/{epochs} {'-'*15}]")
        total_loss = 0
        
        for _ in tqdm(range(num_batches), desc="Đang Train (Batches)", unit="batch", leave=False):
            loss = agent.train_step(replay_buffer, batch_size)
            total_loss += loss
            
        avg_loss = total_loss / num_batches
        print(f"Loss trung bình: {avg_loss:.4f}")
        
        print("Đang đánh giá trên tập Test...")
        metrics = RecSysEvaluator.evaluate_epoch(agent, test_data, train_data, all_vids)
        current_ndcg = metrics['NDCG']
        
        # In nhóm Xếp hạng (Độ chính xác)
        print(f"[Xếp hạng] HR@10: {metrics['HR']:.4f} | NDCG@10: {current_ndcg:.4f} | MAP@10: {metrics['MAP']:.4f}")
        
        # In nhóm Trải nghiệm (Độ phủ & Đa dạng)
        print(f"[Khám phá] Coverage: {metrics['Coverage']*100:.2f}% | ILD: {metrics['ILD']:.4f} | Novelty: {metrics['Novelty']:.4f}")
        
        if current_ndcg > best_ndcg:
            print(f"🌟 NDCG tăng từ {best_ndcg:.4f} lên {current_ndcg:.4f}. Đang lưu mô hình...")
            best_ndcg = current_ndcg
            patience_counter = 0
            torch.save(agent.q_net.state_dict(), model_save_path)
        else:
            patience_counter += 1
            print(f"⚠️ NDCG không tăng. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n🛑 KÍCH HOẠT EARLY STOPPING TẠI EPOCH {epoch+1}!")
                break

    print(f"\n✅ Hoàn tất! Mô hình xuất sắc nhất đã được lưu tại: {os.path.abspath(model_save_path)}")

if __name__ == "__main__":
    run_benchmark()