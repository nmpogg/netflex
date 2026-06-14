import pandas as pd
import numpy as np

def load_and_preprocess(log_path, seq_len=3, train_ratio=0.8):
    print("Đang tải và xử lý dữ liệu KuaiRand...")
    
    # 1. Gọi thêm đầy đủ các cột tương tác sâu
    columns = [
        'user_id', 'video_id', 'time_ms', 'is_click', 'is_like', 
        'is_follow', 'is_comment', 'is_forward', 'is_hate', 
        'play_time_ms', 'duration_ms'
    ]
    df = pd.read_csv(log_path, usecols=columns)
    
    # Lọc bỏ data nhiễu
    df = df[df['duration_ms'] > 0]
    
    # ---------------------------------------------------------
    # BƯỚC 1: TÍNH TOÁN HÀM PHẦN THƯỞNG ĐA TẦNG (Multi-tier Reward)
    # ---------------------------------------------------------
    df['watch_ratio'] = (df['play_time_ms'] / df['duration_ms']).clip(upper=1.5)
    
    # a. Thưởng xem video (Dùng Logarit để làm mượt, tránh việc video quá dài cướp hết điểm)
    r_watch = 2.0 * np.log1p(df['watch_ratio']) 
    
    # b. Thưởng tương tác sâu (Deep Engagement)
    r_engage = (5.0 * df['is_like']) + (10.0 * df['is_comment']) + \
               (10.0 * df['is_forward']) + (15.0 * df['is_follow'])
               
    # c. Phạt (Penalties)
    # Phát hiện Skip nhanh: Có click nhưng xem dưới 2 giây (2000 ms)
    is_quick_skip = ((df['is_click'] == 1) & (df['play_time_ms'] < 2000)).astype(int)
    r_penalty = (20.0 * df['is_hate']) + (5.0 * is_quick_skip)
    
    # Tổng hợp r_t
    df['reward'] = r_watch + r_engage - r_penalty

    # ---------------------------------------------------------
    # BƯỚC 2: TẠO VECTOR HÀNH VI CHO TRẠNG THÁI (Context-Aware State)
    # ---------------------------------------------------------
    # Thay vì chỉ lưu video_id, ta gói thêm thái độ của người dùng với video đó
    df['behavior_context'] = df[['watch_ratio', 'is_like', 'is_comment', 'is_hate']].values.tolist()

    # Sắp xếp lại chuẩn theo chuỗi thời gian
    df = df.sort_values(by=['user_id', 'time_ms'])
    
    trajectories = []
    all_videos = df['video_id'].unique().tolist()
    
    for user_id, user_df in df.groupby('user_id'):
        video_seq = user_df['video_id'].tolist()
        reward_seq = user_df['reward'].tolist()
        behavior_seq = user_df['behavior_context'].tolist() # Lấy thêm mảng hành vi
        
        if len(video_seq) <= seq_len: continue
            
        for i in range(len(video_seq) - seq_len):
            # Xây dựng State: Thay vì 1 list [ID1, ID2, ID3], 
            # Giờ là 1 list các dictionary chứa cả ID và Behavior
            state_history = [
                {"video_id": video_seq[j], "behavior": behavior_seq[j]} 
                for j in range(i, i + seq_len)
            ]
            
            next_state_history = [
                {"video_id": video_seq[j], "behavior": behavior_seq[j]} 
                for j in range(i + 1, i + seq_len + 1)
            ]
            
            trajectories.append({
                'user_id': user_id,
                'state_history': state_history,             # Đã nâng cấp!
                'action_video': video_seq[i + seq_len],
                'reward': reward_seq[i + seq_len],          # Đã nâng cấp!
                'next_state_history': next_state_history,   # Đã nâng cấp!
                'done': 1 if (i == len(video_seq) - seq_len - 1) else 0
            })
            
    mdp_df = pd.DataFrame(trajectories)
    
    split_index = int(len(mdp_df) * train_ratio)
    train_data = mdp_df.iloc[:split_index]
    test_data = mdp_df.iloc[split_index:]
    
    print(f"Tổng số tương tác: Train={len(train_data)}, Test={len(test_data)}")
    return train_data, test_data, all_videos

# Chạy thử để kiểm tra format mới
# train, test, vids = load_and_preprocess_advanced('KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv')
# print(train.iloc[0]['state_history'])