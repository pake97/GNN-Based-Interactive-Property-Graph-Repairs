from dataclasses import dataclass
from typing import Dict, List, Set, Callable, Tuple, Optional
from heapq import nlargest
from collections import deque
import argparse
import polars as pl
import numpy as np
import joblib
import pandas as pd
import warnings
import json
import requests

from openai import OpenAI
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def llm_repair(grdg_node):
    
    
    #grdg_node=grdg_nodes[grdg_nodes['id']==node_id].iloc[0]
    repairs = grdg_node['repairs']
    order=grdg_node['order']
    grdg_node_ids=grdg_node['nodes']
    nodess=[]
    
    for node in json.loads(grdg_node_ids):
    
        #can we keep only columsn not null? 
        nn= graph.filter(pl.col("_id")==node).to_pandas().iloc[0].to_dict()
        #clean null 
        nn = {k: v for k, v in nn.items() if pd.notnull(v) and k not in ['isViolation','violationId']}
        nodess.append(nn)
    constraint=""
    f1s=[0]
    if order == '[0,3,4,6,1,2,5]':
        constraint="MATCH (d:Drug)-[p:PRESCRIBED]-(t:Therapy)-[r:RECEIVED]-(c:Case)-[f:FALLS_UNDER]->(ag:AgeGroup {ageGroup: 'Child'}) RETURN DISTINCT d, t, c, ag, p, r, f"
        f1s=[1,0.5,0.5,0.4,0.4,0.4,0]
    elif (order == "[1,2,0]" or order == "[0,2,1]") and "FALLS_UNDER" in repairs:
        constraint=" MATCH (c:Case)-[r1:FALLS_UNDER]->(a1:AgeGroup), (c)-[r2:FALLS_UNDER]->(a2:AgeGroup) WHERE a1 <> a2 RETURN DISTINCT c, a1, a2, r1, r2"
        f1s=[1,0.5,0]
    elif order == "[0]":
        constraint="MATCH (d)-[r:IS_PRIMARY_SUSPECT]->(d) return d,r"
        f1s=[1]
    else:
        constraint="MATCH (c:Case)-[primary:IS_PRIMARY_SUSPECT]->(d:Drug)<-[secondary:IS_SECONDARY_SUSPECT]-(c) RETURN DISTINCT c, d, primary, secondary"
        f1s=[1,0.5,0]
    
    
    prompt = f"""
    We have the following contraint {constraint}
    causing the following violation with these nodes: {nodess}
    The possible repairs are {repairs}
    Please choose the best repair returning only a string containing the index (from 0 to len(repairs)-1) of the repair that you would choose.
    return only the index of the repair you would choose, no explanation.
    """
    
    f1_gpt = 0
    
    model= "gpt-5.2"

    client = OpenAI()  # uses OPENAI_API_KEY env var

    
    response = client.responses.create(
        model=model,
        instructions=f"Return only one character between: {order}. No explanation.",
        input=prompt,
        max_output_tokens=16,
    )

    try:
        chosen =int(response.output_text.strip())
        f1_gpt = f1s[chosen]
    except:
        f1_gpt = 0
        
    model ="llama3"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0  # makes output deterministic
            }
        }
    )

    result = response.json()["response"].strip()

    # optional cleanup to ensure only 0/1/2
    try:
        chosen = int(result)
        f1_llm = f1s[chosen]
    except:
        f1_llm = 0

    return f1_gpt, f1_llm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    parser.add_argument('--theta', type=float, required=True, help='Theta value (float)')
    parser.add_argument('--budget', type=float, required=True, help='DIfficulty metric')
    parser.add_argument('--metric', type=str, required=True, help='DIfficulty metric')
    parser.add_argument('--dt', type=float, required=True, help='Theta value (float)')
    args = parser.parse_args()
    dataset = args.dataset
    global graph
    global grdg_nodes
    graph = None
    grdg_nodes = None
    if dataset == "faers":
        graph = pl.read_csv("./data/faers/faers_with_violations.csv",low_memory=False)
        grdg_nodes=pl.read_csv("./data/faers/grdg_nodes.csv",low_memory=False)
    if dataset == "finbench":
        graph = pl.read_csv("./data/finbench/finbench_with_violations.csv",low_memory=False)
        grdg_nodes=pl.read_csv("./data/finbench/grdg_nodes.csv",low_memory=False)    
    if dataset == "icij":    
        graph = pl.read_csv("./data/icij/paradise_with_violations.csv",low_memory=False)
        grdg_nodes=pl.read_csv("./data/icij/grdg_nodes.csv",low_memory=False)
    if dataset == "snb":    
        graph = pl.read_csv("./data/snb/snb_with_violations.csv",low_memory=False)
        grdg_nodes=pl.read_csv("./data/snb/grdg_nodes.csv",low_memory=False)
    args = parser.parse_args()
    
    
    theta = args.theta
    budget = args.budget
    metric = args.metric
    dt = args.dt
    
    # Read CSV with polars
    # df_nodes = pl.read_ipc(f"{dataset}/grdgs/{str(theta)}_nodes.feather")
    user_model = joblib.load('super_repair_type_model.pkl')
    
    
    
    # df_file = pl.read_ipc(f"{dataset}/gap_{metric}_{str(theta)}_{str(budget)}/best_assign.feather")
    # print(df_file.height)
    
    #filter df_nodes to only include nodes with difficulty < theta
    
    df_nodes = grdg_nodes.filter(pl.col("difficulty") < theta)
    print("Number of nodes to evaluate: ", df_nodes.height)
    
    
    
    #randomly pick 100 
    
    df_file = df_nodes.sample(n=100)
    
    #get avg difficulty 
    avg_difficulty = df_nodes.select(pl.col("difficulty").mean()).to_numpy()[0][0]
    
    f1_map = {0:1,2: 0.5,1: 0.66,3: 0}  
    
    full_llm_f1 = []
    full_gpt_f1 = []
    full_cost = 0
    
    
    
    half_llm_f1 = []
    half_gpt_f1 = []
    half_cost = 0
    
    for node in df_nodes.iter_rows(named=True):
    
        
        # real_difficulty = df_nodes.filter(pl.col("id")==node).select('real_difficulty').to_numpy()[0][0]
        # difficulty = df_nodes.filter(pl.col("id")==node).select('difficulty').to_numpy()[0][0]
        real_difficulty = node['real_difficulty']
        difficulty = node['difficulty']
        
        
         
        f1_gpt, f1_llm = llm_repair(node)
        
        full_llm_f1.append(f1_llm)  # LLM would fix
        full_gpt_f1.append(f1_gpt)  # GPT would fix
        
        full_cost += 0
        
        
        
        if difficulty > avg_difficulty:     
            res = f1_map[user_model.predict([[0.9, real_difficulty]])[0]]
            half_gpt_f1.append(res)
            half_llm_f1.append(res)
            half_cost += 1
        else: 
            half_gpt_f1.append(f1_gpt)  # GPT would fix
            half_llm_f1.append(f1_llm)  # LLM would fix
            half_cost += 0
    

    
    full_mean_f1_llm = np.mean(full_llm_f1) if len(full_llm_f1) > 0 else None
    full_mean_f1_gpt = np.mean(full_gpt_f1) if len(full_gpt_f1) > 0 else None
    half_mean_f1_llm = np.mean(half_llm_f1) if len(half_llm_f1) > 0 else None
    half_mean_f1_gpt = np.mean(half_gpt_f1) if len(half_gpt_f1) > 0 else None
    
    

    results = {
    "full_avg_f1_llm": full_mean_f1_llm,
    "full_avg_f1_gpt": full_mean_f1_gpt,
    "full_cost": full_cost,
    "half_avg_f1_llm": half_mean_f1_llm,
    "half_avg_f1_gpt": half_mean_f1_gpt,
    "half_cost": half_cost
    }

    with open(f"{dataset}/gap_{metric}_{str(dt)}_{str(budget)}/llm_NEW_quality_report.json", "w") as f:
        json.dump(results, f, indent=4)


    
