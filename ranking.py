from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast
import json
import numpy as np
from scipy.stats import entropy



for dataset in ["snb", "icij", "faers", "finbench"]:
    print(f"Processing dataset: {dataset}")
    df = pd.read_csv("./data/"+dataset+"/grdg_nodes.csv", low_memory=False)

    alpha=0.505985
    beta=0.195751


    df['num_nodes'] = df['nodes'].apply(lambda x: len(json.loads(x)))
    df['num_edges'] = df['edges'].apply(lambda x: len(ast.literal_eval(x)))

    # df['real_difficulty']= alpha*df['sum_edge_score']/df['num_nodes'] + beta*df['sum_node_conf']/df['num_edges']
    # df['sum_sum_diff'] = df['sum_edge_score'] + df['sum_node_conf']
    # df['avg_sum_diff']= 0.5*df['sum_edge_score'] + 0.5*df['sum_node_conf']
    # df['sum_avg_difficulty']= df['sum_edge_score']/df['num_nodes'] + df['sum_node_conf']/df['num_edges']


    # List of methods to compare against real difficulty
    methods = ['degree', 'pagerank', 'cs_cl', 'difficulty', "real_difficulty", "cognitive_load"]
    correlations = {}

    for method in methods:
        # Compute Spearman rank correlation
        rho, p_value = spearmanr(df['real_difficulty'], df[method])
        correlations[method] = {'rho': rho, 'p_value': p_value}

    # Convert to DataFrame for display or plotting
    corr_df = pd.DataFrame(correlations).T.reset_index().rename(columns={
        'index': 'method', 'rho': 'Spearman ρ', 'p_value': 'p-value'
    })

    # Display
    print(corr_df)

    # # Optional: visualize
    # plt.figure(figsize=(6, 4))
    # sns.barplot(data=corr_df, x='method', y='Spearman ρ')
    # plt.title('Spearman Correlation with Real Difficulty')
    # plt.ylim(-1, 1)
    # plt.axhline(0, color='gray', linestyle='--')
    # plt.tight_layout()
    # plt.show()

    from scipy.stats import kendalltau

    kendall_results = {}

    for method in methods:
        tau, p_value = kendalltau(df['real_difficulty'], df[method])
        kendall_results[method] = {'tau': tau, 'p_value': p_value}

    kendall_df = pd.DataFrame(kendall_results).T.reset_index().rename(columns={
        'index': 'method', 'tau': "Kendall's τ", 'p_value': 'p-value'
    })

    print("\nKendall's τ correlation:")
    print(kendall_df)


for dataset in ["snb", "icij", "faers", "finbench"]:
    print(f"Processing dataset: {dataset}")
    df = pd.read_csv("./data/"+dataset+"/grdg_nodes.csv", low_memory=False)
    methods = ['degree', 'pagerank', 'cs_cl', 'difficulty', "real_difficulty"]
    alpha=0.505985
    beta=0.195751
    df['num_nodes'] = df['nodes'].apply(lambda x: len(json.loads(x)))
    df['num_edges'] = df['edges'].apply(lambda x: len(ast.literal_eval(x)))
    
    
    df['real_difficulty'] = None
    for row in df.itertuples():
        if row.num_nodes > 0 and row.num_edges > 0:
            df.at[row.Index, 'real_difficulty'] = alpha * row.sum_edge_score / row.num_nodes + beta * row.sum_node_conf / row.num_edges
        elif row.num_nodes == 0 and row.num_edges > 0:
            df.at[row.Index, 'real_difficulty'] = beta * row.sum_node_conf / row.num_edges
        elif row.num_nodes > 0 and row.num_edges == 0:
            df.at[row.Index, 'real_difficulty'] = alpha * row.sum_edge_score / row.num_nodes
    
    print(df['real_difficulty'].mean(), df['real_difficulty'].std())
    print(df['difficulty'].mean(), df['difficulty'].std())
    for method in methods:
        min_value = df[method].min()
        max_value = df[method].max()
        if(not method=='real_difficulty') or not method=='difficulty':
            df[method] = (df[method] - min_value) / (max_value - min_value)
        
        mean_squared_error = ((df['real_difficulty'] - df[method]) **2).mean()
        print(f"Mean Squared Error between real_difficulty and {method} for {dataset}: {mean_squared_error}")
        
        

for dataset in ["snb", "icij", "faers", "finbench"]:
    print(f"Processing dataset: {dataset}")
    df = pd.read_csv("./data/"+dataset+"/grdg_nodes.csv", low_memory=False)
    methods = ['degree', 'pagerank', 'cs_cl', 'difficulty', "real_difficulty"]
    alpha=0.505985
    beta=0.195751
    df['num_nodes'] = df['nodes'].apply(lambda x: len(json.loads(x)))
    df['num_edges'] = df['edges'].apply(lambda x: len(ast.literal_eval(x)))
    df['real_difficulty'] = None
    for row in df.itertuples():
        if row.num_nodes > 0 and row.num_edges > 0:
            df.at[row.Index, 'real_difficulty'] = alpha * row.sum_edge_score / row.num_nodes + beta * row.sum_node_conf / row.num_edges
        elif row.num_nodes == 0 and row.num_edges > 0:
            df.at[row.Index, 'real_difficulty'] = beta * row.sum_node_conf / row.num_edges
        elif row.num_nodes > 0 and row.num_edges == 0:
            df.at[row.Index, 'real_difficulty'] = alpha * row.sum_edge_score / row.num_nodes
     
    print(df['real_difficulty'].mean(), df['real_difficulty'].std())
    print(df.columns)
    for method in methods: 
        
        # Ensure the columns are numpy arrays
        p = df['real_difficulty'].dropna().values
        q = df[method].dropna().values

        # Normalize to ensure they are valid probability distributions
        p = p / p.sum()
        q = q / q.sum()

        # Add small epsilon to avoid log(0) or division by zero (if needed)
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)

        # Compute KL divergence: D_KL(P || Q)
        kl_divergence = entropy(p, q)

        print(f"KL Divergence (real diff || {method}): {kl_divergence}")        



    