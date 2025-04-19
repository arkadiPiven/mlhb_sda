import torch
import torch.nn as nn


class PairwiseHingeRankingLoss(nn.Module):
    def __init__(self, lower_is_better: bool = True):
        """
        A smooth surrogate for inversion count.
        If lower_is_better=True, then smaller `targets` are more relevant,
        so you want score[i] > score[j] whenever target[i] < target[j].
        """
        super().__init__()
        self.lower_is_better = lower_is_better
        self.sigmoid = nn.Sigmoid()

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ensure batch dimension
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            targets = targets.unsqueeze(0)

        # scores: (B, N), targets: (B, N)
        s_i = scores.unsqueeze(2)   # (B, N, 1)
        s_j = scores.unsqueeze(1)   # (B, 1, N)
        t_i = targets.unsqueeze(2)  # (B, N, 1)
        t_j = targets.unsqueeze(1)  # (B, 1, N)

        # build the mask of “true” pairs:
        # if lower_is_better, i ≻ j when t_i < t_j; else when t_i > t_j
        if self.lower_is_better:
            pref_mask = (t_i < t_j).float()
        else:
            pref_mask = (t_i > t_j).float()

        # probability of an inversion: score_j > score_i
        inv_prob = self.sigmoid(s_j - s_i)

        # only count inversions on the true‐preference pairs
        loss_mat = pref_mask * inv_prob
        num_pairs = pref_mask.sum().clamp_min(1.0)

        return loss_mat.sum() / num_pairs

def kld(self, vec_true, vec_compare):
    ind = vec_true.data * vec_compare.data > 0
    ind_var = chainer.Variable(ind)
    include_nan = vec_true * F.log(vec_true / vec_compare)
    z = chainer.Variable(np.zeros((len(ind), 1), dtype=np.float32))
    # return np.nansum(vec_true * np.log(vec_true / vec_compare))
    return F.sum(F.where(ind_var, include_nan, z))

def jsd(self, vec_true, vec_compare):
    vec_mean = 0.5 * (vec_true + vec_compare)
    return 0.5 * self.kld(vec_true, vec_mean) + 0.5 * self.kld(vec_compare, vec_mean)

def topkprob(self, vec, k=5):
    vec_sort = np.sort(vec)[-1::-1]
    topk = vec_sort[:k]
    ary = np.arange(k)
    return np.prod([np.exp(topk[i]) / np.sum(np.exp(topk[i:])) for i in ary])

def listwise_cost(self, list_ans, list_pred):
    return - np.sum(self.topkprob(list_ans) * np.log(self.topkprob(list_pred)))