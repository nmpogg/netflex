import numpy as np
import torch
from tqdm import tqdm

class RecSysEvaluator:
    @staticmethod
    def get_ranking_metrics(recommended_list, ground_truth, k=10):
        top_k = recommended_list[:k]
        if ground_truth in top_k:
            index = top_k.index(ground_truth)
            hr = 1.0
            ndcg = 1.0 / np.log2(index + 2)
            map_score = 1.0 / (index + 1)
            return hr, ndcg, map_score
        return 0.0, 0.0, 0.0

    @staticmethod
    def compute_popularity_dict(train_data, all_videos):
        counts = train_data['action_video'].value_counts().to_dict()
        total_interactions = len(train_data)
        return {vid: counts.get(vid, 0) / total_interactions for vid in all_videos}

    @classmethod
    def evaluate_epoch(cls, agent, test_data, train_data, all_videos, k=10, num_candidates=100):
        hr_list, ndcg_list, map_list = [], [], []
        ild_list, novelty_list = [], []
        recommended_items_set = set()
        
        popularity_dict = cls.compute_popularity_dict(train_data, all_videos)
        num_tests = len(test_data)
        
        print("\nĐang chuẩn bị Ma trận Đánh giá...")
        
        all_decoys_matrix = np.random.choice(all_videos, size=(num_tests, num_candidates - 1))
        
        with torch.no_grad():
            all_embeddings = agent.q_net.video_embed.weight.detach().cpu().numpy()

        for idx, row in tqdm(enumerate(test_data.itertuples()), total=num_tests, desc="Đang Eval", unit="mẫu", leave=False):
            user_id = row.user_id
            state = row.state_history
            gt_action = row.action_video
            
            candidates = all_decoys_matrix[idx].tolist()
            if gt_action not in candidates: 
                candidates.append(gt_action)
            else:
                candidates.append(np.random.choice(all_videos)) # Đề phòng trùng lặp ngẫu nhiên
            np.random.shuffle(candidates)
            
            # Agent xếp hạng
            ranked_list = agent.rank_videos(user_id, state, candidates)
            top_k_recs = ranked_list[:k]
            
            # Chấm điểm Ranking
            hr, ndcg, mmap = cls.get_ranking_metrics(top_k_recs, gt_action, k)
            hr_list.append(hr)
            ndcg_list.append(ndcg)
            map_list.append(mmap)
            
            # Chấm điểm Novelty & Coverage
            recommended_items_set.update(top_k_recs)
            novelty_scores = [-np.log2(popularity_dict.get(vid, 1e-5)) for vid in top_k_recs]
            novelty_list.append(np.mean(novelty_scores))
            
            if len(top_k_recs) > 1:
                embs = all_embeddings[top_k_recs] # Lookup siêu tốc O(1)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                sim_matrix = np.dot(embs, embs.T) / (norms @ norms.T + 1e-8)
                
                # Chỉ lấy tam giác trên của ma trận (trừ đường chéo)
                idx_triu = np.triu_indices(len(top_k_recs), k=1)
                ild_score = np.mean(1.0 - sim_matrix[idx_triu])
                ild_list.append(ild_score)
            else:
                ild_list.append(0.0)
            
        metrics = {
            "HR": np.mean(hr_list),
            "NDCG": np.mean(ndcg_list),
            "MAP": np.mean(map_list),
            "ILD": np.mean(ild_list),
            "Novelty": np.mean(novelty_list),
            "Coverage": len(recommended_items_set) / len(all_videos)
        }
        
        return metrics