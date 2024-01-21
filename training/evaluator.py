import math
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score

def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2\
        / np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0
    cash_flow=[]
    sharpe_li5 = []
    sharpe_li10 = []
    sharpe_li1 = []
    total_rank=[]
    total_pre_rank=[]
    prefiction=[]
    real_ground=[]
    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            total_rank.append(cur_rank)
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)
        rank_pre = np.argsort(prediction[:, i])

        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            total_pre_rank.append(cur_rank)
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)
        # calculate mrr of top1
        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt
            
        performance['ndcg_score_top5'] = ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1))
        performance['ndcg_score_top10'] = ndcg_score(np.array(list(gt_top10)).reshape(1,-1), np.array(list(pre_top10)).reshape(1,-1))
        
        # back testing on top 1
        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top
        sharpe_li1.append(real_ret_rat_top)
        # back testing on top 5
        real_ret_rat_top5 = 0
        pre_ret_rat_top5 = 0
        real_top5=0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
            pre_ret_rat_top5 += prediction[pre][i]
        real_ret_rat_top5 /= 5
        pre_ret_rat_top5 /=5
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top5)
        prefiction.append(pre_ret_rat_top5)
        for g in gt_top5:
            real_top5+=ground_truth[g][i]
        real_top5/=5
        real_ground.append(real_top5)
        # back testing on top 10
        real_ret_rat_top10 = 0
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
        real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10
        sharpe_li10.append(real_ret_rat_top10)

    performance['mrrt'] = mrr_top / (prediction.shape[1] - all_miss_days_top)
    performance['btl'] = bt_long
    performance['btl5'] = bt_long5
    performance['btl10'] = bt_long10
    performance['sharpe5'] = (np.mean(sharpe_li5)/np.std(sharpe_li5))*15.87 #To annualize
    performance['sharpe10'] = (np.mean(sharpe_li10)/np.std(sharpe_li10))*15.87 #To annualize
    performance['sharpe1'] = (np.mean(sharpe_li1)/np.std(sharpe_li1))*15.87 #To annualize
    performance['data5']=sharpe_li5 #real return
    performance['data1']=gt_top5
    performance['data2'] =pre_top5
    performance['pre'] = prefiction
    #performance['total_pre_rank']=total_pre_rank #print rank
    #performance['total_rank']=total_rank
    performance['prediction']=prediction
    performance['gt']=ground_truth

    return performance