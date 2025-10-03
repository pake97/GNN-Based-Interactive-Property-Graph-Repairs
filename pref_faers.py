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
    
    
    
    
    res = connector.neo4j_Connector.query("MATCH p=(d:Drug)-[p:PRESCRIBED]-(t:Therapy)-[r:RECEIVED]-(c:Case)-[f:FALLS_UNDER]->(ag:AgeGroup {ageGroup: 'Child'}) where f.toDelete=True RETURN count(p) as total")
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
    res = connector.neo4j_Connector.query("MATCH p=(d:Drug)-[p:PRESCRIBED]-(t:Therapy)-[r:RECEIVED]-(c:Case)-[f:FALLS_UNDER]->(ag:AgeGroup {ageGroup: 'Child'}) where p.toDelete=True RETURN count(p) as total")
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
    res = connector.neo4j_Connector.query("MATCH p=(a2:AgeGroup)<-[r2:FALLS_UNDER]-(c:Case)-[r1:FALLS_UNDER]->(a1:AgeGroup) WHERE a1 <> a2 and r1.toDelete=True RETURN count(p) as total")
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
    res = connector.neo4j_Connector.query("MATCH p=(a2:AgeGroup)<-[r2:FALLS_UNDER]-(c:Case)-[r1:FALLS_UNDER]->(a1:AgeGroup) WHERE a1 <> a2 and r2.toDelete=True RETURN count(p) as total")
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
    res = connector.neo4j_Connector.query("MATCH p=(d:Drug)-[r:IS_PRIMARY_SUSPECT]->(d:Drug) where r.toDelete=True RETURN count(p) as total")
    num = res[0]['total']
    pd_tp+=num
    pd_fp+=0
    pd_fn+=0
    pu_tp+=0
    pu_fp+=0
    pu_fn+=num
    ps_tp+=0
    ps_fp+=0
    ps_fn+=0
    res = connector.neo4j_Connector.query("MATCH p=(c:Case)-[primary:IS_PRIMARY_SUSPECT]->(d:Drug)<-[secondary:IS_SECONDARY_SUSPECT]-(c) where secondary.toDelete=True RETURN count(p)")
    num = res[0]['total']
    pd_tp+=0.5*num
    pd_fp+=0.5*num
    pd_fn+=0.5*num
    pu_tp+=0
    pu_fp+=0
    pu_fn+=num
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