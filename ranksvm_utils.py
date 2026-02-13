import numpy as np
from sklearn.metrics import ndcg_score

def calculate_ranking_ndcg(y_true, y_pred, groups):
    """
    Calculate NDCG properly for ranking problems.
    Parameters:
    y_true (array-like): True relevance scores
    y_pred (array-like): Predicted relevance scores
    groups (array-like): Group sizes (number of items per query/session)
    Returns:
    float: Average NDCG score across all groups
    """
    ndcg_scores = []
    start_idx = 0
    for group_size in groups:
        end_idx = start_idx + group_size
    if group_size > 1: # Only calculate NDCG for groups with multiple items
        group_true = y_true[start_idx:end_idx]
        group_pred = y_pred[start_idx:end_idx]
    # Reshape for ndcg_score (expects 2D)
    ndcg = ndcg_score([group_true], [group_pred])
    ndcg_scores.append(ndcg)
    start_idx = end_idx
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def get_group_sizes(y_df):
    """
    Get group sizes from a dataframe containing the group column.
    Parameters:
    y_df (pd.DataFrame): Dataframe with group column
    Returns:
    pd.Series: Group sizes indexed by group
    """
    return y_df.reset_index().groupby("group")['group'].count()