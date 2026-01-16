import os
import json
import csv
import argparse





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')

    
    args = parser.parse_args()
    
    dataset = args.dataset

    rows = []
    header = ["dataset", "metric", "theta", "budget", "avg_f1", "num_repairs", "total_cost","unique_users","certified01","certified05","combined_f1"]



    f1_gnn = {1.0: (0.0, 0)}

    for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        with open(f"{dataset}/grdgs/{str(theta)}_f1.txt", "r") as f:
            lines = f.readlines()
            f1 = float(lines[0].strip().split(": ")[1])
            deleted = int(lines[2].strip().split(": ")[1])
            f1_gnn[theta] = (f1, deleted)



    path = "{dataset}".format(dataset=dataset)

    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and 'difficulty' in name and 'gap' in name:
            try:
                
                
                certificates = {}
                try:
                    with open(os.path.join(full_path, "summary.json"), "r") as f:
                        certificates = json.load(f)
                except FileNotFoundError:
                    print(f"summary.json not found in {full_path}, skipping certificates.")
                    certificates = {
                        "certified01": False,
                        "certified05": False
                    }
                with open(os.path.join(full_path, "quality_report.json"), "r") as f:
                    #gap_difficulty_1.0_200.0    
                    theta = float(name.split("_")[2])
                    budget = float(name.split("_")[3])
                    metric = name.split("_")[1]
                    
                    data = json.load(f)
                    
                    
                    
                    row = [
                        dataset,
                        metric,
                        theta,
                        budget,
                        data.get("avg_f1"),
                        data.get("num_repairs"),
                        data.get("total_cost"),
                        data.get("unique_users"),
                        certificates.get("certified01", False),
                        certificates.get("certified05", False), 
                        (f1_gnn.get(theta)[0]*f1_gnn.get(theta)[1] + data.get("avg_f1")*data.get("num_repairs"))/(f1_gnn.get(theta)[1]+data.get("num_repairs")) 
                    ]
                    
                    
                    rows.append(row)
            except FileNotFoundError:
                print(f"quality_report.json not found in {full_path}, skipping.")
                continue
        # Salva la tabella in CSV
        with open(f"{dataset}/exp3_quality_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    

    print("Results saved in quality_summary.csv")
