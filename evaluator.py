import numpy as np

class RecSysEvaluator:
    @staticmethod
    def get_metrics(recommended_list, ground_truth, k=10):
        top_k = recommended_list[:k]
        if ground_truth in top_k:
            index = top_k.index(ground_truth)
            hr = 1.0
            ndcg = 1.0 / np.log2(index + 2)
            map_score = 1.0 / (index + 1)
            return hr, ndcg, map_score
        return 0.0, 0.0, 0.0

    @classmethod
    def evaluate_epoch(cls, agent, test_data, all_videos, k=10, num_candidates=100):
        hr_list, ndcg_list, map_list = [], [], []
        
        for _, row in test_data.iterrows():
            state = row['state_history']
            gt_action = row['action_video']
            
            # Trộn Ground Truth với 99 video mồi nhử (Decoys)
            candidates = list(np.random.choice(all_videos, num_candidates - 1))
            if gt_action not in candidates: candidates.append(gt_action)
            np.random.shuffle(candidates)
            
            ranked_list = agent.rank_videos(state, candidates)
            hr, ndcg, mmap = cls.get_metrics(ranked_list, gt_action, k)
            
            hr_list.append(hr); ndcg_list.append(ndcg); map_list.append(mmap)
            
        return np.mean(hr_list), np.mean(ndcg_list), np.mean(map_list)