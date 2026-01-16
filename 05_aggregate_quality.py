import os
import json
import csv
import argparse





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    parser.add_argument('--theta', type=float, required=True, help='Theta value (float)')
    parser.add_argument('--budget', type=float, required=True, help='DIfficulty metric')    
    
    args = parser.parse_args()
    
    dataset = args.dataset
    theta = args.theta
    budget = args.budget
    
    rows = []
    header = ["dataset", "metric", "theta", "budget", "avg_f1", "unique_users"]

    for metric in ["difficulty", "normalized_cs_cl", "normalized_degree", "normalized_pagerank"]:
    
    
        data = {}
        with open(f"{dataset}/gap_{metric}_{str(theta)}_{str(budget)}/quality_report.json", "r") as f:
            data = json.load(f)
    
        row = [
                    dataset,
                    metric,
                    theta,
                    budget,
                    data.get("avg_f1"),
                    data.get("max_f1"),
                    data.get("min_f1"),
                    data.get("std_f1"),
                    data.get("num_repairs"),
                    data.get("total_cost"),
                    data.get("unique_users"),
                ]
        rows.append(row)
    # Salva la tabella in CSV
    with open(f"{dataset}/exp1_quality_summary_{str(theta)}_{str(budget)}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        
    

    print("Results saved in quality_summary.csv")
