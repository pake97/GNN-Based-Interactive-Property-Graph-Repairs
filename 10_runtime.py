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
    header = ["dataset", "metric", "theta", "budget","iterations","time1","time2","total_time"]



    path = "{dataset}".format(dataset=dataset)

    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and 'difficulty' in name and 'gap' in name:
            try:
                
                
                certificates = {}
                try:
                    with open(os.path.join(full_path, "summary.json"), "r") as f:
                        certificates = json.load(f)
                        #gap_difficulty_1.0_200.0    
                        theta = float(name.split("_")[2])
                        budget = float(name.split("_")[3])
                        metric = name.split("_")[1]
                        

                        if certificates.get("iterations", 1)==0:
                            certificates["iterations"]=1
                        
                        row = [
                            dataset,
                            metric,
                            theta,
                            budget,
                            certificates.get("iterations", 0),
                            certificates.get("time1", 0),
                            certificates.get("time2", 0), 
                            certificates.get("time2", 0)*certificates.get("iterations", 1) +certificates.get("time1", 0)
                        ]
                        
                        
                        rows.append(row)
                except FileNotFoundError:
                    print(f"summary.json not found in {full_path}, skipping certificates.")
                    
                    
            except FileNotFoundError:
                print(f"quality_report.json not found in {full_path}, skipping.")
                continue
        # Salva la tabella in CSV
        with open(f"{dataset}/exp4_runtime_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    

    print("Results saved in quality_summary.csv")
