import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ---------------------------------------------------------
# 1. Mạng Q-Network Nâng cao (Context-Aware & Temporal Weighing)
# ---------------------------------------------------------
class AdvancedRecSysQNetwork(nn.Module):
    def __init__(self, num_videos, video_embed_dim=64, behavior_dim=4, seq_len=3, hidden_dim=256):
        super(AdvancedRecSysQNetwork, self).__init__()
        self.seq_len = seq_len
        self.video_embed_dim = video_embed_dim
        self.behavior_dim = behavior_dim
        
        # Lớp nhúng dùng chung cho cả Video trong lịch sử và Video ứng viên
        self.video_embed = nn.Embedding(num_videos, video_embed_dim)
        
        # Kéo phẳng chuỗi: Mỗi bước có (video_embed_dim + behavior_dim) đặc trưng
        flat_history_dim = (video_embed_dim + behavior_dim) * seq_len
        
        # Lớp này đóng vai trò tự học trọng số thời gian (thay thế cho hàm Mean Pooling)
        self.history_aggregation = nn.Linear(flat_history_dim, 128)
        self.ln_state = nn.LayerNorm(128)
        
        # Mạng MLP chấm điểm Q tổng hợp: State (128) + Action (video_embed_dim)
        self.fc1 = nn.Linear(128 + video_embed_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, history_ids, history_behaviors, candidate_ids):
        # history_ids: (batch_size, seq_len)
        # history_behaviors: (batch_size, seq_len, behavior_dim)
        # candidate_ids: (batch_size, 1)
        batch_size = history_ids.size(0)
        
        # 1. Trích xuất Embedding cho lịch sử xem phim
        hist_embeds = self.video_embed(history_ids) # (batch_size, seq_len, video_embed_dim)
        
        # 2. Nối chuỗi đặc trưng Video với Vector hành vi thực tế từ log
        # Kết quả: (batch_size, seq_len, video_embed_dim + behavior_dim)
        hist_features = torch.cat([hist_embeds, history_behaviors], dim=-1)
        
        # 3. Kéo phẳng chuỗi thời gian để đưa qua lớp học trọng số vị trí/thời gian
        hist_flat = hist_features.view(batch_size, -1) 
        user_state = torch.relu(self.ln_state(self.history_aggregation(hist_flat))) # (batch_size, 128)
        
        # 4. Trích xuất Embedding cho Video ứng viên (Action)
        cand_embeds = self.video_embed(candidate_ids).squeeze(1) # (batch_size, video_embed_dim)
        
        # 5. Khâu nối State + Action và đưa vào MLP tính toán Q-value
        x = torch.cat([user_state, cand_embeds], dim=-1) # (batch_size, 128 + video_embed_dim)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)

# ---------------------------------------------------------
# 2. Replay Buffer giữ nguyên cơ chế lưu trữ
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(list, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------
# 3. Agent DQN xử lý dữ liệu cấu trúc phức tạp
# ---------------------------------------------------------
class AdvancedDQNAgent:
    def __init__(self, num_videos, video_embed_dim=64, behavior_dim=4, seq_len=3, lr=1e-3, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = AdvancedRecSysQNetwork(num_videos, video_embed_dim, behavior_dim, seq_len).to(self.device)
        self.target_net = AdvancedRecSysQNetwork(num_videos, video_embed_dim, behavior_dim, seq_len).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _parse_batch_state(self, batch_state):
        """Hàm bổ trợ bóc tách danh sách từ điển dạng [{'video_id': x, 'behavior': [...]}] thành Tensor"""
        batch_ids = []
        batch_behaviors = []
        for history in batch_state:
            ids = [item['video_id'] for item in history]
            behaviors = [item['behavior'] for item in history]
            batch_ids.append(ids)
            batch_behaviors.append(behaviors)
        return torch.LongTensor(batch_ids).to(self.device), torch.FloatTensor(batch_behaviors).to(self.device)

    def select_action(self, state, candidate_actions):
        # Trả về hành động có Q-value cao nhất (Hàm này có thể tích hợp trực tiếp vào hàm rank_videos bên dưới)
        return self.rank_videos(state, candidate_actions)[0]

    def rank_videos(self, state, candidate_actions):
        """Giao tiếp với Evaluator để sắp xếp danh sách phim ứng viên"""
        self.q_net.eval()
        with torch.no_grad():
            # Tạo batch giả lập cho cấu trúc dữ liệu nâng cao
            single_id, single_beh = self._parse_batch_state([state])
            hist_ids = single_id.repeat(len(candidate_actions), 1)
            hist_behs = single_beh.repeat(len(candidate_actions), 1, 1)
            cand_ids = torch.LongTensor(candidate_actions).unsqueeze(1).to(self.device)
            
            q_values = self.q_net(hist_ids, hist_behs, cand_ids).squeeze(-1).cpu().numpy()
            
        ranked_indices = np.argsort(q_values)[::-1]
        self.q_net.train()
        return [candidate_actions[i] for i in ranked_indices]

    def train_step(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return 0.0

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

        # Bóc tách cấu trúc dữ liệu lồng nhau phức tạp từ log
        hist_ids, hist_behs = self._parse_batch_state(state_batch)
        next_hist_ids, next_hist_behs = self._parse_batch_state(next_state_batch)
        
        actions = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # 1. Tính giá trị Q hiện tại
        current_q = self.q_net(hist_ids, hist_behs, actions)

        # 2. Tính giá trị Q mục tiêu sử dụng Target Network
        with torch.no_grad():
            # Trong môi trường Offline RL, action kế tiếp lấy trực tiếp từ dữ liệu log thực tế để đảm bảo tính an toàn
            next_actions = actions # Mô phỏng hành động kế tiếp có sẵn trong log mẫu
            next_q = self.target_net(next_hist_ids, next_hist_behs, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 3. Cập nhật Gradient mạng chính
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Cập nhật mềm tham số sang mạng Target
        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()