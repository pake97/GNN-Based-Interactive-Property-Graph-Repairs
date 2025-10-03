import threading
import random
import time
import math
import numpy as np
from utils.neo4j_connector import Neo4jConnector

def main(dataset):
    #print("Clearing dataset")
    connector = neo4j_Connector = Neo4jConnector()
    connector.clearNeo4j()
    #print("Loading dataset")
    connector.loadDatasetToNeo4j(dataset)
    
    
    pd_tp=0
    pd_fp=0
    pd_fn=0
    pu_tp=0
    pu_fp=0
    pu_fn=0
    ps_tp=0
    ps_fp=0
    ps_fn=0
    
    
    
    res = connector.query("MATCH path=(p:Person {id: id})-[r:Guarantee]->(p) r.toDelete = true return count(path) as total")
    num = res[0]['total']
    pd_tp+=num
    pd_fp+=0
    pd_fn+=0
    pu_tp+=0
    pu_fp+=0
    pu_fn+=num
    ps_tp+=num
    ps_fp+=0
    ps_fn+=0
    
    ran = 0.02 + random.random() * (0.1 - 0.02)
    res = connector.query("MATCH path=(c:Company)-[r:Apply]->(l:Loan) WHERE l.interestRate < 0.02 and r.toDelete = false and l.interestRate_GT = "+str(ran)+"RETURN count(path) as total")
    num = res[0]['total']
    pd_tp+=0
    pd_fp+=0
    pd_fn+=num
    pu_tp+=num
    pu_fp+=0
    pu_fn+=0
    ps_tp+=num
    ps_fp+=0
    ps_fn+=0
    res = connector.query("MATCH path=(c:Company)-[r:Apply]->(l:Loan) WHERE l.interestRate < 0.02 and r.toDelete = false and l.interestRate_GT <> "+str(ran)+"RETURN count(path) as total")
    num = res[0]['total']
    pd_tp+=0
    pd_fp+=0
    pd_fn+=num
    pu_tp+=num
    pu_fp+=0
    pu_fn+=0
    ps_tp+=num
    ps_fp+=0
    ps_fn+=0


    res = connector.query("MATCH path=(guarantor:Person)-[s:Guarantee]->(guaranteed:Person)-[r:Own]->(account:Account) WHERE guarantor <> guaranteed AND account.balance < 500000000 and s.toDelete = True RETURN count(path) as total")
    num = res[0]['total']
    pd_tp+=0.5*num
    pd_fp+=0.5*num
    pd_fn+=0.5*num
    pu_tp+=0
    pu_fp+=0
    pu_fn+=num
    ps_tp+=0.5*num
    ps_fp+=0.5*num
    ps_fn+=0.5*num
    
    value = 500_000_001 + int(random.random() * (1_000_000_000 - 500_000_001))
    res = connector.query("MATCH path=(guarantor:Person)-[s:Guarantee]->(guaranteed:Person)-[r:Own]->(account:Account) WHERE guarantor <> guaranteed AND account.balance < 500000000 and account.balance_GT = "+str(value)+" RETURN count(path) as total")
    num = res[0]['total']
    pd_tp+=0
    pd_fp+=0
    pd_fn+=num
    pu_tp+=num
    pu_fp+=0
    pu_fn+=0
    ps_tp+=num
    ps_fp+=0
    ps_fn+=0
    res = connector.query("MATCH path=(guarantor:Person)-[s:Guarantee]->(guaranteed:Person)-[r:Own]->(account:Account) WHERE guarantor <> guaranteed AND account.balance < 500000000 and account.balance_GT <> "+str(value)+" RETURN count(path) as total")
    num = res[0]['total']
    pd_tp+=0
    pd_fp+=0
    pd_fn+=num
    pu_tp+=num
    pu_fp+=0
    pu_fn+=0
    ps_tp+=num
    ps_fp+=0
    ps_fn+=0
    
    
    
    
    res = connector.query("MATCH path=(a:Account)-[r:Transfer]->(a1:Account) WHERE a.isBlocked = true and r.toDelete=true RETURN count(path) as total")
    num = res[0]['total']
    pd_tp+=0.5*num
    pd_fp+=0.5*num
    pd_fn+=0.5*num
    pu_tp+=0
    pu_fp+=0
    pu_fn+=num
    ps_tp+=0
    ps_fp+=0
    ps_fn+=num
    
    res = connector.query("MATCH path=(a:Account)-[r:Transfer]->(a1:Account) WHERE a.isBlocked = true and r.toDelete=false RETURN count(path) as total")
    num = res[0]['total']
    pd_tp+=0
    pd_fp+=0
    pd_fn+=num
    pu_tp+=num
    pu_fp+=0
    pu_fn+=0
    ps_tp+=num
    ps_fp+=0
    ps_fn+=0
    
    ps_precision = ps_tp / (ps_tp + ps_fp) if (ps_tp + ps_fp) > 0 else 0
    ps_recall = ps_tp / (ps_tp + ps_fn) if (ps_tp + ps_fn) > 0 else 0
    ps_f1_score = 2 * (ps_precision * ps_recall) / (ps_precision + ps_recall) if (ps_precision + ps_recall) > 0 else 0

    print(f"Preferred Schema F1 Score: {ps_f1_score:.6f}")

    pu_precision = pu_tp / (pu_tp + pu_fp) if (pu_tp + pu_fp) > 0 else 0
    pu_recall = pu_tp / (pu_tp + pu_fn) if (pu_tp + pu_fn) > 0 else 0
    pu_f1_score = 2 * (pu_precision * pu_recall) / (pu_precision + pu_recall) if (pu_precision + pu_recall) > 0 else 0
    
    print(f"Preferred Update F1 Score: {pu_f1_score:.6f}")
    
    pd_precision = pd_tp / (pd_tp + pd_fp) if (pd_tp + pd_fp) > 0 else 0
    pd_recall = pd_tp / (pd_tp + pd_fn) if (pd_tp + pd_fn) > 0 else 0
    pd_f1_score = 2 * (pd_precision * pd_recall) / (pd_precision + pd_recall) if (pd_precision + pd_recall) > 0 else 0
    
    print(f"Preferred Delete F1 Score: {pd_f1_score:.6f}")
    
# Example Usage
if __name__ == "__main__":
    
    main('healtchare')