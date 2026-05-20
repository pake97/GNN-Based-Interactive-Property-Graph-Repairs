from dataclasses import dataclass
from typing import Dict, List, Set, Callable, Tuple, Optional
from heapq import nlargest
from collections import deque
import argparse
import polars as pl
import numpy as np
import joblib
import warnings
import json
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="X does not have valid feature names")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    parser.add_argument('--theta', type=float, required=True, help='Theta value (float)')
    parser.add_argument('--budget', type=float, required=True, help='DIfficulty metric')
    parser.add_argument('--metric', type=str, required=True, help='DIfficulty metric')
    
    
    args = parser.parse_args()
    
    dataset = args.dataset
    theta = args.theta
    budget = args.budget
    metric = args.metric
    
    # Read CSV with polars
    df_nodes = pl.read_ipc(f"{dataset}/grdgs/{str(theta)}_nodes.feather")
    user_model = joblib.load('super_repair_type_model.pkl')
    df_users = pl.read_ipc(f"./{dataset}/users.feather")
    
    
    df_file = pl.read_ipc(f"{dataset}/gap_{metric}_{str(theta)}_{str(budget)}/best_assign.feather")
  
    
    f1_map = {0:1,2: 0.5,1: 0.66,3: 0}  
    # tpr_map={0:1,2: 1,1: 1,3: 0}
    # fpr_map={0:0,2: 1,1: 0.5,3: 0.5}
    # Higher = better answer
    relevance_map = {
        0: 3,   # best
        2: 2,
        1: 1,
        3: 0    # worst
    }

    rows = []
    f1 = []
    rrs = []
    kendalls = []
    for assignment in df_file.iter_rows(named=True):
        user = assignment['user_id']
        node = assignment['id']
        cost = assignment['cost']
       
        
        real_difficulty = df_nodes.filter(pl.col("id")==node).select('real_difficulty').to_numpy()[0][0]
        
        
        skill = df_users.filter(pl.col("user_id")==user).select("skills").to_numpy()[0][0]        
        
        probs = user_model.predict_proba([[skill, real_difficulty]])[0]
        prediction = user_model.predict([[skill, real_difficulty]])[0]

        pred_ranking = np.argsort(-probs)

        # MRR: best answer is option 0
        correct_item = 0
        rank = np.where(pred_ranking == correct_item)[0][0] + 1
        rr = 1.0 / rank

        # F1 of selected prediction
        res_f1 = f1_map[prediction]

        # Kendall Tau between predicted probabilities and true relevance
        option_ids = np.arange(len(probs))
        true_relevance = np.array([relevance_map[o] for o in option_ids])

        tau, p_value = kendalltau(probs, true_relevance)

        f1.append(res_f1)
        rrs.append(rr)
        kendalls.append(tau)

        # Store one row per option
        for option_id, prob in enumerate(probs):
            rows.append({
                "user_id": user,
                "node_id": node,
                "cost": cost,
                "skill": skill,
                "real_difficulty": real_difficulty,

                "option_id": option_id,
                "predicted_prob": prob,

                # For ROC-AUC: pretend best answer is positive
                "is_best_answer": 1 if option_id == 0 else 0,

                # For ranking metrics
                "true_relevance": relevance_map[option_id],

                # Prediction-level info repeated for convenience
                "predicted_answer": prediction,
                "predicted_rank": int(np.where(pred_ranking == option_id)[0][0] + 1),
                "rr": rr,
                "f1": res_f1,
                "kendall_tau": tau,
            })

    df_eval = pl.DataFrame(rows)
    y_true = df_eval["is_best_answer"].to_numpy()
    y_score = df_eval["predicted_prob"].to_numpy()

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    
    # plt.figure(figsize=(6, 5))
    # plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    # plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve: Best Answer vs Rest")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    unique_users = df_file.select("user_id").unique()
    total_cost = df_file.select(pl.col("cost").sum()).to_numpy()[0][0] 
    
    results = {
    "avg_f1": float(np.mean(f1)),
    "mean_kendall_tau": np.nanmean(kendalls),
    "roc_auc": auc,
    "avg_rr": float(np.mean(rrs)),
    "num_repairs": int(len(f1)),
    "total_cost": float(total_cost),
    "unique_users": int(unique_users.height),
    }

    with open(f"{dataset}/gap_{metric}_{str(theta)}_{str(budget)}/new_quality_report.json", "w") as f:
        json.dump(results, f, indent=4)
    
    
    