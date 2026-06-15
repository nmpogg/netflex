import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class HybridRecSysQNetwork(nn.Module):
    def __init__(self, num_videos, num_authors, num_musics, 
                 video_dim=64, author_dim=16, music_dim=16, 
                 behavior_dim=4, seq_len=3, user_dim=6, hidden_dim=256):
        super(HybridRecSysQNetwork, self).__init__()
        
        self.video_embed = nn.Embedding(num_videos, video_dim)
        self.author_embed = nn.Embedding(num_authors, author_dim)
        self.music_embed = nn.Embedding(num_musics, music_dim)
        
        self.full_video_dim = video_dim + author_dim + music_dim + 1
        flat_history_dim = (self.full_video_dim + behavior_dim) * seq_len
        
        self.history_aggregation = nn.Linear(flat_history_dim, 128)
        self.ln_state = nn.LayerNorm(128)
        
        # Concat Điểm: Dynamic State (128) + Static Persona (user_dim) + Candidate Video (full_video_dim)
        self.fc1 = nn.Linear(128 + user_dim + self.full_video_dim, hidden_dim)
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

    def forward(self, hist_ids, hist_behs, hist_feats, user_feats, cand_ids, cand_feats):
        # 1. Tạo Dynamic State từ chuỗi phim
        hist_full_embeds = self._get_full_video_representation(hist_ids, hist_feats)
        hist_combined = torch.cat([hist_full_embeds, hist_behs], dim=-1)
        hist_flat = hist_combined.view(hist_ids.size(0), -1) 
        dynamic_state = torch.relu(self.ln_state(self.history_aggregation(hist_flat)))
        
        # 2. Tạo Action Vector
        cand_full_embeds = self._get_full_video_representation(cand_ids, cand_feats).squeeze(1)
        
        # 3. Tính Q-value
        x = torch.cat([dynamic_state, user_feats, cand_full_embeds], dim=-1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, user_id, state, action, reward, next_state, done):
        self.buffer.append((user_id, state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(list, zip(*batch))
        
    def __len__(self): 
        return len(self.buffer)


class HybridDQNAgent:
    def __init__(self, video_feat_dict, user_feat_dict, num_videos, num_authors, num_musics, num_users, user_dim=6, lr=1e-3, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"-> Khởi chạy Model trên nền tảng: {self.device}")

        # Nạp bảng Video Features lên VRAM
        self.video_table = torch.zeros((num_videos, 3), device=self.device)
        for vid, feats in video_feat_dict.items():
            if vid < num_videos:
                self.video_table[vid] = torch.tensor(feats, device=self.device)
                
        # Nạp bảng User Features lên VRAM
        self.user_table = torch.zeros((num_users, user_dim), device=self.device)
        for uid, feats in user_feat_dict.items():
            if uid < num_users:
                self.user_table[uid] = torch.tensor(feats, device=self.device)

        self.q_net = HybridRecSysQNetwork(num_videos, num_authors, num_musics, user_dim=user_dim).to(self.device)
        self.target_net = HybridRecSysQNetwork(num_videos, num_authors, num_musics, user_dim=user_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _parse_state(self, batch_user_ids, batch_state):
        batch_ids, batch_behs = [], []
        for history in batch_state:
            batch_ids.append([item['video_id'] for item in history])
            batch_behs.append([item['behavior'] for item in history])
            
        ids_tensor = torch.LongTensor(batch_ids).to(self.device)
        behs_tensor = torch.FloatTensor(batch_behs).to(self.device)
        feats_tensor = self.video_table[ids_tensor]
        
        user_ids_tensor = torch.LongTensor(batch_user_ids).to(self.device)
        user_feats_tensor = self.user_table[user_ids_tensor]
        
        return ids_tensor, behs_tensor, feats_tensor, user_feats_tensor

    def rank_videos(self, user_id, state, candidate_actions):
        self.q_net.eval()
        with torch.no_grad():
            hist_ids, hist_behs, hist_feats, user_feats = self._parse_state([user_id], [state])
            
            num_cands = len(candidate_actions)
            hist_ids = hist_ids.repeat(num_cands, 1)
            hist_behs = hist_behs.repeat(num_cands, 1, 1)
            hist_feats = hist_feats.repeat(num_cands, 1, 1)
            user_feats = user_feats.repeat(num_cands, 1)
            
            cand_ids = torch.LongTensor(candidate_actions).unsqueeze(1).to(self.device)
            cand_feats = self.video_table[cand_ids]
            
            q_values = self.q_net(hist_ids, hist_behs, hist_feats, user_feats, cand_ids, cand_feats).squeeze(-1).cpu().numpy()
            
        ranked_indices = np.argsort(q_values)[::-1]
        self.q_net.train()
        return [candidate_actions[i] for i in ranked_indices]

    def train_step(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size: return 0.0

        user_b, state_b, action_b, reward_b, next_state_b, done_b = replay_buffer.sample(batch_size)

        hist_ids, hist_behs, hist_feats, user_feats = self._parse_state(user_b, state_b)
        next_hist_ids, next_hist_behs, next_hist_feats, next_user_feats = self._parse_state(user_b, next_state_b)
        
        actions = torch.LongTensor(action_b).unsqueeze(1).to(self.device)
        action_feats = self.video_table[actions]
        rewards = torch.FloatTensor(reward_b).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(done_b).unsqueeze(1).to(self.device)

        current_q = self.q_net(hist_ids, hist_behs, hist_feats, user_feats, actions, action_feats)

        with torch.no_grad():
            next_q = self.target_net(next_hist_ids, next_hist_behs, next_hist_feats, next_user_feats, actions, action_feats)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()