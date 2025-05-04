import torch 
import torch.nn.functional as F


def generate_rank_tensor(ranked_list, candidates_list, pad_value=-1):
  
    rank_indices = torch.zeros(len(ranked_list), dtype=torch.int64)
    for rank, item in enumerate(ranked_list):
        item_index = candidates_list.index(item)
        rank_indices[rank] = item_index
        
    max_k= len(candidates_list)

    if rank_indices.size(0) < max_k:
        # Calculate how much padding is needed
        pad_size = max_k - rank_indices.size(0)
        # Pad the rank_indices tensor with the given pad_value
        padding = torch.full(( pad_size, ), pad_value, dtype=rank_indices.dtype)
        # Concatenate the original rank with the padding
        rank_indices = torch.cat([rank_indices, padding], dim=0)

    return rank_indices


def eval(rank, labels, ks, pad_value=-1):
    
    rank = rank.unsqueeze(0) 
    
    metrics = {}
    labels = F.one_hot(labels, num_classes=rank.size(1))
    answer_count = labels.sum(1)
    labels_float = labels.float()
    len_seq = rank.shape[-1]

    for k in sorted(ks, reverse=True):
        if len_seq >= k:
            cut = rank[:, :k]      

            # Replace pad_value (-1) with 0 (or any valid index) temporarily
            cut_for_gather = cut.clone()
            # Replace invalid index
            cut_for_gather[cut_for_gather == pad_value] = 0  
            hits = labels_float.gather(1, cut_for_gather)
            # Set hits corresponding to padding elements (pad_value) back to zero
            hits[cut == pad_value] = 0

            # MRR calculation
            metrics['MRR@%d' % k] = \
                (hits / torch.arange(1, k+1).unsqueeze(0).to(
                    labels.device)).sum(1).mean().cpu().item()
            
            # Hit Ratio calculation
            hit_ratio = (hits.sum(1) > 0).float().mean().cpu().item()
            metrics['HR@%d' % k] = hit_ratio
            
            # NDCG calculation
            position = torch.arange(2, 2+k)
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights.to(hits.device)).sum(1)
            idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                                for n in answer_count]).to(dcg.device)
            ndcg = (dcg / idcg).mean()
            metrics['NDCG@%d' % k] = ndcg.cpu().item()
            
        else:
            # If len_seq < k, set metrics to 0
            metrics['MRR@%d' % k] = 0.0
            metrics['HR@%d' % k] = 0.0
            metrics['NDCG@%d' % k] = 0.0

    return metrics

