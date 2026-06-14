import numpy as np

class RecSysEvaluator:
    @staticmethod
    def get_hit_rate(recommended_list, ground_truth, k=10):
        """Hit Rate @ K: Có trúng video người dùng thực sự xem trong Top K không?"""
        top_k = recommended_list[:k]
        return 1.0 if ground_truth in top_k else 0.0

    @staticmethod
    def get_ndcg(recommended_list, ground_truth, k=10):
        """NDCG @ K: Tính điểm dựa trên vị trí xếp hạng. Càng ở trên cao điểm càng lớn."""
        top_k = recommended_list[:k]
        if ground_truth in top_k:
            index = top_k.index(ground_truth)
            return 1.0 / np.log2(index + 2) # Cộng 2 vì index bắt đầu từ 0
        return 0.0

    @staticmethod
    def get_map(recommended_list, ground_truth, k=10):
        """MAP @ K (Mean Average Precision) cho trường hợp 1 ground truth."""
        top_k = recommended_list[:k]
        if ground_truth in top_k:
            index = top_k.index(ground_truth)
            return 1.0 / (index + 1)
        return 0.0

    @classmethod
    def evaluate_epoch(cls, agent, test_data, candidate_videos, k=10):
        """Chạy đánh giá toàn bộ tập Test và trả về số điểm trung bình."""
        hr_list, ndcg_list, map_list = [], [], []
        
        for index, row in test_data.iterrows():
            state = row['state_history']
            ground_truth_action = row['action_video']
            
            # Giả lập: Lấy Ground Truth trộn với 99 video ngẫu nhiên khác để tạo tập 100 candidates
            # Trong thực tế, candidate_videos được tạo sẵn cho từng user_id
            current_candidates = list(np.random.choice(candidate_videos, 99))
            if ground_truth_action not in current_candidates:
                current_candidates.append(ground_truth_action)
                
            np.random.shuffle(current_candidates)
            
            # Yêu cầu Agent xếp hạng danh sách candidates này
            ranked_list = agent.rank_videos(state, current_candidates)
            
            hr_list.append(cls.get_hit_rate(ranked_list, ground_truth_action, k))
            ndcg_list.append(cls.get_ndcg(ranked_list, ground_truth_action, k))
            map_list.append(cls.get_map(ranked_list, ground_truth_action, k))
            
        return np.mean(hr_list), np.mean(ndcg_list), np.mean(map_list)