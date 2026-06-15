import numpy as np
import torch

class RecSysEvaluator:
    @staticmethod
    def get_ranking_metrics(recommended_list, ground_truth, k=10):
        """Chấm điểm Hit Rate, NDCG, MAP"""
        top_k = recommended_list[:k]
        if ground_truth in top_k:
            index = top_k.index(ground_truth)
            hr = 1.0
            ndcg = 1.0 / np.log2(index + 2)
            map_score = 1.0 / (index + 1)
            return hr, ndcg, map_score
        return 0.0, 0.0, 0.0

    @staticmethod
    def get_ild(agent, top_k_list):
        """Intra-List Diversity: Khoảng cách Cosine trung bình giữa các video trong Top K"""
        if len(top_k_list) < 2: 
            return 0.0
            
        # Trích xuất linh hồn của video (64-dim Vector) trực tiếp từ não của DQN
        with torch.no_grad():
            tensor_ids = torch.LongTensor(top_k_list).to(agent.device)
            embeddings = agent.q_net.video_embed(tensor_ids).cpu().numpy()
            
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                v1, v2 = embeddings[i], embeddings[j]
                norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm_product == 0:
                    sim = 0.0
                else:
                    sim = np.dot(v1, v2) / norm_product
                distances.append(1.0 - sim) # Distance = 1 - Cosine Similarity
                
        return np.mean(distances)

    @staticmethod
    def get_novelty(top_k_list, popularity_dict):
        """Novelty: Tính bằng Self-Information của xác suất xuất hiện"""
        novelty_scores = []
        for vid in top_k_list:
            # Nếu video chưa từng xuất hiện, cho xác suất cực nhỏ để tránh lỗi log(0)
            p_i = popularity_dict.get(vid, 1e-5) 
            novelty_scores.append(-np.log2(p_i))
        return np.mean(novelty_scores)

    @staticmethod
    def compute_popularity_dict(train_data, all_videos):
        """Hàm bổ trợ: Tính xác suất xuất hiện của từng video trong tập Train"""
        # Đếm số lần xuất hiện của từng action_video
        counts = train_data['action_video'].value_counts().to_dict()
        total_interactions = len(train_data)
        
        popularity_dict = {}
        for vid in all_videos:
            popularity_dict[vid] = counts.get(vid, 0) / total_interactions
        return popularity_dict

    @classmethod
    def evaluate_epoch(cls, agent, test_data, train_data, all_videos, k=10, num_candidates=100):
        # 1. Khởi tạo mảng lưu trữ
        hr_list, ndcg_list, map_list = [], [], []
        ild_list, novelty_list = [], []
        
        # Tập hợp chứa TẤT CẢ các video đã từng được Agent mang ra gợi ý
        recommended_items_set = set()
        
        # Tính toán từ điển độ phổ biến (Chỉ tính 1 lần mỗi epoch)
        popularity_dict = cls.compute_popularity_dict(train_data, all_videos)
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            state = row['state_history']
            gt_action = row['action_video']
            
            # Tạo 100 Candidates (1 True + 99 Decoys)
            candidates = list(np.random.choice(all_videos, num_candidates - 1))
            if gt_action not in candidates: 
                candidates.append(gt_action)
            np.random.shuffle(candidates)
            
            # Lấy Top K xếp hạng từ Agent
            ranked_list = agent.rank_videos(user_id, state, candidates)
            top_k_recs = ranked_list[:k]
            
            # --- CHẤM ĐIỂM NHÓM 1: RANKING ---
            hr, ndcg, mmap = cls.get_ranking_metrics(top_k_recs, gt_action, k)
            hr_list.append(hr)
            ndcg_list.append(ndcg)
            map_list.append(mmap)
            
            # --- CHẤM ĐIỂM NHÓM 2: TRẢI NGHIỆM ---
            recommended_items_set.update(top_k_recs) # Cập nhật độ phủ
            ild_list.append(cls.get_ild(agent, top_k_recs))
            novelty_list.append(cls.get_novelty(top_k_recs, popularity_dict))
            
        # Tổng hợp các chỉ số trung bình
        metrics = {
            "HR": np.mean(hr_list),
            "NDCG": np.mean(ndcg_list),
            "MAP": np.mean(map_list),
            "ILD": np.mean(ild_list),
            "Novelty": np.mean(novelty_list),
            "Coverage": len(recommended_items_set) / len(all_videos) # Tỷ lệ % video được mang ra ánh sáng
        }
        
        return metrics