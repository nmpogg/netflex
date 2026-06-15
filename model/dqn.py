import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class HybridRecSysQNetwork(nn.Module):
    def __init__(self, num_videos, num_authors, num_musics, 
                 video_dim=64, author_dim=16, music_dim=16, 
                 behavior_dim=4, seq_len=3, hidden_dim=256):
        super(HybridRecSysQNetwork, self).__init__()
        
        self.video_embed = nn.Embedding(num_videos, video_dim)
        self.author_embed = nn.Embedding(num_authors, author_dim)
        self.music_embed = nn.Embedding(num_musics, music_dim)
        
        # Đặc trưng 1 Video = ID(64) + Author(16) + Music(16) + Duration(1) = 97
        self.full_video_dim = video_dim + author_dim + music_dim + 1
        
        # State = (97 + 4 hành vi) * seq_len
        flat_history_dim = (self.full_video_dim + behavior_dim) * seq_len
        self.history_aggregation = nn.Linear(flat_history_dim, 128)
        self.ln_state = nn.LayerNorm(128)
        
        self.fc1 = nn.Linear(128 + self.full_video_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)

    def _get_full_video_representation(self, video_ids, video_features):
        v_emb = self.video_embed(video_ids)
        duration = video_features[..., 0:1].float()
        author_idx = video_features[..., 1].long()
        music_idx = video_features[..., 2].long()
        
        a_emb = self.author_embed(author_idx)
        m_emb = self.music_embed(music_idx)
        return torch.cat([v_emb, a_emb, m_emb, duration], dim=-1)

    def forward(self, hist_ids, hist_behs, hist_feats, cand_ids, cand_feats):
        # 1. Xử lý Lịch sử
        hist_full_embeds = self._get_full_video_representation(hist_ids, hist_feats)
        hist_combined = torch.cat([hist_full_embeds, hist_behs], dim=-1)
        
        hist_flat = hist_combined.view(hist_ids.size(0), -1) 
        user_state = torch.relu(self.ln_state(self.history_aggregation(hist_flat)))
        
        # 2. Xử lý Video Ứng viên
        cand_full_embeds = self._get_full_video_representation(cand_ids, cand_feats).squeeze(1)
        
        # 3. Tính Q
        x = torch.cat([user_state, cand_full_embeds], dim=-1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(list, zip(*batch))
    def __len__(self): return len(self.buffer)

class HybridDQNAgent:
    def __init__(self, feature_dict, num_videos, num_authors, num_musics, lr=1e-3, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"-> Agent đang khởi chạy trên: {self.device}")

        # Nạp feature_dict vào Tensor tĩnh trên GPU để truy xuất siêu tốc (O(1))
        self.feature_table = torch.zeros((num_videos, 3), device=self.device)
        for vid, feats in feature_dict.items():
            if vid < num_videos:
                self.feature_table[vid] = torch.tensor(feats, device=self.device)

        self.q_net = HybridRecSysQNetwork(num_videos, num_authors, num_musics).to(self.device)
        self.target_net = HybridRecSysQNetwork(num_videos, num_authors, num_musics).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _parse_state(self, batch_state):
        batch_ids, batch_behs = [], []
        for history in batch_state:
            batch_ids.append([item['video_id'] for item in history])
            batch_behs.append([item['behavior'] for item in history])
        
        ids_tensor = torch.LongTensor(batch_ids).to(self.device)
        behs_tensor = torch.FloatTensor(batch_behs).to(self.device)
        feats_tensor = self.feature_table[ids_tensor] # Truy xuất tính năng siêu tốc
        return ids_tensor, behs_tensor, feats_tensor

    def rank_videos(self, state, candidate_actions):
        self.q_net.eval()
        with torch.no_grad():
            hist_ids, hist_behs, hist_feats = self._parse_state([state])
            
            # Nhân bản lịch sử ra N dòng để map với N video ứng viên
            num_cands = len(candidate_actions)
            hist_ids = hist_ids.repeat(num_cands, 1)
            hist_behs = hist_behs.repeat(num_cands, 1, 1)
            hist_feats = hist_feats.repeat(num_cands, 1, 1)
            
            cand_ids = torch.LongTensor(candidate_actions).unsqueeze(1).to(self.device)
            cand_feats = self.feature_table[cand_ids]
            
            q_values = self.q_net(hist_ids, hist_behs, hist_feats, cand_ids, cand_feats).squeeze(-1).cpu().numpy()
            
        ranked_indices = np.argsort(q_values)[::-1]
        self.q_net.train()
        return [candidate_actions[i] for i in ranked_indices]

    def train_step(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size: return 0.0

        state_b, action_b, reward_b, next_state_b, done_b = replay_buffer.sample(batch_size)

        hist_ids, hist_behs, hist_feats = self._parse_state(state_b)
        next_hist_ids, next_hist_behs, next_hist_feats = self._parse_state(next_state_b)
        
        actions = torch.LongTensor(action_b).unsqueeze(1).to(self.device)
        action_feats = self.feature_table[actions]
        rewards = torch.FloatTensor(reward_b).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(done_b).unsqueeze(1).to(self.device)

        current_q = self.q_net(hist_ids, hist_behs, hist_feats, actions, action_feats)

        with torch.no_grad():
            next_q = self.target_net(next_hist_ids, next_hist_behs, next_hist_feats, actions, action_feats)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()