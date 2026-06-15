import pandas as pd
import numpy as np

def load_video_features(feature_path):
    print("1. Đang tải Video Features (Action Context)...")
    df_feat = pd.read_csv(feature_path)
    
    # Chuẩn hóa thời lượng
    max_duration = df_feat['video_duration'].max()
    df_feat['duration_norm'] = df_feat['video_duration'] / (max_duration + 1e-5)
    
    # Mã hóa nhãn (Label Encoding)
    df_feat['author_code'] = df_feat['author_id'].astype('category').cat.codes
    df_feat['music_code'] = df_feat['music_id'].astype('category').cat.codes
    
    video_feature_dict = {}
    for _, row in df_feat.iterrows():
        vid = int(row['video_id'])
        video_feature_dict[vid] = [row['duration_norm'], row['author_code'], row['music_code']]
        
    num_videos = int(df_feat['video_id'].max()) + 1
    num_authors = len(df_feat['author_code'].unique())
    num_musics = len(df_feat['music_code'].unique())
    
    return video_feature_dict, num_videos, num_authors, num_musics


def load_user_features(feature_path):
    print("2. Đang tải User Features (Static Persona)...")
    df_user = pd.read_csv(feature_path)
    
    cols_to_norm = ['follow_user_num', 'fans_user_num', 'register_days']
    cols_binary = ['is_live_streamer', 'is_video_author', 'is_lowactive_period']
    
    # Chuẩn hóa Min-Max
    for col in cols_to_norm:
        df_user[col] = df_user[col] / (df_user[col].max() + 1e-5)
        
    user_feature_dict = {}
    for _, row in df_user.iterrows():
        uid = int(row['user_id'])
        user_feature_dict[uid] = [row[c] for c in cols_binary + cols_to_norm]
        
    num_users = int(df_user['user_id'].max()) + 1
    user_feature_dim = len(cols_binary + cols_to_norm)
    
    return user_feature_dict, num_users, user_feature_dim


def load_logs_and_build_mdp(log_path, seq_len=3, train_ratio=0.8):
    print("3. Đang tải Log và xây dựng Markov Decision Process...")
    cols = ['user_id', 'video_id', 'time_ms', 'is_click', 'is_like', 
            'is_follow', 'is_comment', 'is_forward', 'is_hate', 'play_time_ms', 'duration_ms']
    df = pd.read_csv(log_path, usecols=cols)
    df = df[df['duration_ms'] > 0]
    
    # Reward Shaping
    df['watch_ratio'] = (df['play_time_ms'] / df['duration_ms']).clip(upper=1.5)
    r_watch = 2.0 * np.log1p(df['watch_ratio']) 
    r_engage = (5.0 * df['is_like']) + (10.0 * df['is_comment']) + (10.0 * df['is_forward']) + (15.0 * df['is_follow'])
    is_quick_skip = ((df['is_click'] == 1) & (df['play_time_ms'] < 2000)).astype(int)
    r_penalty = (20.0 * df['is_hate']) + (5.0 * is_quick_skip)
    
    df['reward'] = r_watch + r_engage - r_penalty
    df['behavior_context'] = df[['watch_ratio', 'is_like', 'is_comment', 'is_hate']].values.tolist()
    
    df = df.sort_values(by=['user_id', 'time_ms'])
    trajectories = []
    all_interactions_videos = df['video_id'].unique().tolist()
    
    for user_id, user_df in df.groupby('user_id'):
        video_seq = user_df['video_id'].tolist()
        reward_seq = user_df['reward'].tolist()
        behavior_seq = user_df['behavior_context'].tolist()
        
        if len(video_seq) <= seq_len: continue
            
        for i in range(len(video_seq) - seq_len):
            state_history = [{"video_id": video_seq[j], "behavior": behavior_seq[j]} for j in range(i, i + seq_len)]
            next_state_history = [{"video_id": video_seq[j], "behavior": behavior_seq[j]} for j in range(i + 1, i + seq_len + 1)]
            
            trajectories.append({
                'user_id': user_id,
                'state_history': state_history,
                'action_video': video_seq[i + seq_len],
                'reward': reward_seq[i + seq_len],
                'next_state_history': next_state_history,
                'done': 1 if (i == len(video_seq) - seq_len - 1) else 0
            })
            
    mdp_df = pd.DataFrame(trajectories)
    split_index = int(len(mdp_df) * train_ratio)
    train_data = mdp_df.iloc[:split_index]
    test_data = mdp_df.iloc[split_index:]
    
    print(f"   -> Hoàn tất! Train: {len(train_data)} mẫu | Test: {len(test_data)} mẫu")
    return train_data, test_data, all_interactions_videos