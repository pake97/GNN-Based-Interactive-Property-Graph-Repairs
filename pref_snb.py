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
    
    









  




    
    res = connector.query("MATCH path = (po:Post)<-[c:CREATED]-(p:Person)-[l:LIKES]-(po) where c.toDelete_GT =True return count(path) as total")
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
    res = connector.query("MATCH path = (po:Post)<-[c:CREATED]-(p:Person)-[l:LIKES]-(po) where l.toDelete_GT =True return count(path) as total")
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
    
    res = connector.query("MATCH path = (pl1:Place)<-[l:LIVES_IN]-(p:Person)-[:WORK_AT]->(o:Organisation)-[ln:LOCATED_IN]->(pl:Place) WHERE id(pl) <> id(pl1) and l.toDelete_GT = true return count(path) as total")
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
    res = connector.query("MATCH path = (pl1:Place)<-[l:LIVES_IN]-(p:Person)-[:WORK_AT]->(o:Organisation)-[ln:LOCATED_IN]->(pl:Place) WHERE id(pl) <> id(pl1) and ln.toDelete_GT = true return count(path) as total")
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
    
    
    
    
    res = connector.query("MATCH path = (p:Person)<-[m:HAS_MEMBER]-(f:Forum) WHERE p.age < f.ageRequirement and f.ageRequirement_GT = 16  return count(path) as total")
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
    value = random.randint(1, 10)
    res = connector.query("MATCH path = (p:Person)<-[m:HAS_MEMBER]-(f:Forum) WHERE p.age < f.ageRequirement and p.age_GT = f.ageRequirement + "+str(value)+"  return count(path) as total")
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
    
    
    res = connector.query("MATCH path = (c:Comment)-[r:TO]->(c1:Post) WHERE c.creationMillis < c1.creationMillis  return count(path) as tota")
    num = res[0]['total']
    pd_tp+=0
    pd_fp+=0
    pd_fn+=num
    pu_tp+=0.5*num
    pu_fp+=0.5*num
    pu_fn+=0.5*num
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
    
    main('snbIngestion')