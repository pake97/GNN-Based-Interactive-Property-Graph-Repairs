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
    header = ["dataset", "metric", "theta", "budget", "avg_f1", "num_repairs", "total_cost","unique_users","certified01","certified05","combined_f1","combined_f1_1","combined_f1_2","combined_f1_3"]



    f1_gnn = {1.0: (0.0, 0)}
    f1_gnn_1={1.0: (0.0, 0)}
    f1_gnn_2={1.0: (0.0, 0)}
    f1_gnn_3={1.0: (0.0, 0)}
    for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        with open(f"{dataset}/grdgs/{str(theta)}_f1.txt", "r") as f:
            
            lines = f.readlines()
            if "GNN" in lines[0]:
                f1 = float(lines[3].strip().split(": ")[1])
                f1_1 = float(lines[0].strip().split(": ")[1])
                f1_2 = float(lines[1].strip().split(": ")[1])
                f1_3 = float(lines[2].strip().split(": ")[1])
            
                deleted = int(lines[4].strip().split(": ")[1])
                deleted_1 = int(lines[5].strip().split(": ")[1])
                deleted_2 = int(lines[6].strip().split(": ")[1])
                deleted_3 = int(lines[7].strip().split(": ")[1])

                f1_gnn[theta] = (f1, deleted)
                f1_gnn_1[theta] = (f1_1, deleted_1)
                f1_gnn_2[theta] = (f1_2, deleted_2)
                f1_gnn_3[theta] = (f1_3, deleted_3)
            else: 
                f1 = float(lines[0].strip().split(": ")[1])
                deleted = int(lines[2].strip().split(": ")[1])
                f1_gnn[theta] = (f1, deleted)            


    path = "{dataset}".format(dataset=dataset)

    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        print(f"Processing {full_path}...")
        if os.path.isdir(full_path) and 'normalized' in name and 'gap' in name:
            print(f"Found directory: {full_path}")
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
                    theta = float(name.split("_")[3])
                    budget = float(name.split("_")[4])
                    metric = name.split("_")[1]
                    
                    budgets=[]
                    if dataset=="faers":
                        
                        budgets=[0,50, 100,200, 264, 529, 793, 1058, 1322, 1586, 1851, 2115, 2380, 2644.0]
                    else:
                        budgets=[0.0, 250.0, 500.0, 750.0, 1134.0, 2269.0, 3403.0, 4538.0, 5672.0, 6806.0, 7941.0, 9075.0, 10210.0, 11344.0] 
                    
                    if budget in budgets:
                        data = json.load(f)
                        print(data)
                        
                        header = ["dataset", "metric", "theta", "budget", "avg_f1", "num_repairs", "total_cost","unique_users", "1_avg_f1", "1_num_repairs", "1_total_cost","1_unique_users", "2_avg_f1", "2_num_repairs", "2_total_cost","2_unique_users", "3_avg_f1", "3_num_repairs", "3_total_cost","3_unique_users","certified01","certified05","combined_f1","combined_f1_1","combined_f1_2","combined_f1_3"]
                        
                        
                        print(f1_gnn, f1_gnn_1, f1_gnn_2, f1_gnn_3)
                        combined_f1=None
                        combined_f1_1=None
                        combined_f1_2=None
                        combined_f1_3=None
                        
                        if f1_gnn.get(theta)[1]>0:
                            combined_f1 = (f1_gnn.get(theta)[0]*f1_gnn.get(theta)[1] + data.get("avg_f1")*data.get("num_repairs"))/(f1_gnn.get(theta)[1]+data.get("num_repairs"))
                        else:
                            combined_f1 = data.get("avg_f1")
                        
                        if f1_gnn_1.get(theta)[1]>0:
                            combined_f1_1 = (f1_gnn_1.get(theta)[0]*f1_gnn_1.get(theta)[1] + data.get("1_avg_f1")*data.get("1_num_repairs"))/(f1_gnn_1.get(theta)[1]+data.get("1_num_repairs"))
                        else:
                            combined_f1_1 = data.get("1_avg_f1")
                        
                        if f1_gnn_2.get(theta)[1]>0:
                            combined_f1_2 = (f1_gnn_2.get(theta)[0]*f1_gnn_2.get(theta)[1] + data.get("2_avg_f1")*data.get("2_num_repairs"))/(f1_gnn_2.get(theta)[1]+data.get("2_num_repairs"))
                        else:
                            combined_f1_2 = data.get("2_avg_f1")    
                        
                        if f1_gnn_3.get(theta)[1]>0:
                            combined_f1_3 = (f1_gnn_3.get(theta)[0]*f1_gnn_3.get(theta)[1] + data.get("3_avg_f1")*data.get("3_num_repairs"))/(f1_gnn_3.get(theta)[1]+data.get("3_num_repairs"))
                        else:
                            combined_f1_3 = data.get("3_avg_f1")
                     
                        print(f"Combined F1: {combined_f1}, Combined F1 1: {combined_f1_1}, Combined F1 2: {combined_f1_2}, Combined F1 3: {combined_f1_3}")
                        
                        row = [
                            dataset,
                            metric,
                            theta,
                            budget,
                            data.get("avg_f1"),
                            data.get("num_repairs"),
                            data.get("total_cost"),
                            data.get("unique_users"),                    

                            data.get("1_avg_f1"),
                            data.get("1_num_repairs"),
                            data.get("1_total_cost"),
                            data.get("1_unique_users"),                    

                            data.get("2_avg_f1"),
                            data.get("2_num_repairs"),
                            data.get("2_total_cost"),
                            data.get("2_unique_users"),                    

                            data.get("3_avg_f1"),
                            data.get("3_num_repairs"),
                            data.get("3_total_cost"),
                            data.get("3_unique_users"),      
                            certificates.get("certified01", False),
                            certificates.get("certified05", False), 
                            combined_f1,
                            combined_f1_1,
                            combined_f1_2,
                            combined_f1_3
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
