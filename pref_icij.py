import threading
import random
import time
import math
import numpy as np
from utils.neo4j_connector import Neo4jConnector

def main(dataset):
    #print("Clearing dataset")
    connector = neo4j_Connector = Neo4jConnector()
    #connector.clearNeo4j()
    #print("Loading dataset")
    #connector.loadDatasetToNeo4j(dataset)
    
    
    pd_tp=0
    pd_fp=0
    pd_fn=0
    pu_tp=0
    pu_fp=0
    pu_fn=0
    ps_tp=0
    ps_fp=0
    ps_fn=0
    
    


    
    
    

  




    
    res = connector.query("MATCH (a:Address)-[p:REGISTERED_ADDRESS]-(b) WHERE apoc.text.indexOf(b.country_codes, a.country_codes) < 0 and p.toDelete=True return count(a) as total")
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
    res = connector.query("MATCH path=(o)-[of:OFFICER_OF]->(e)-[ra1:REGISTERED_ADDRESS]->(a:Address)<-[ra2:REGISTERED_ADDRESS]-(o:Officer) where ra1.toDelete=True RETURN count(path) as total")
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
    res = connector.query("MATCH path=(o)-[of:OFFICER_OF]->(e)-[ra1:REGISTERED_ADDRESS]->(a:Address)<-[ra2:REGISTERED_ADDRESS]-(o:Officer) where ra2.toDelete=True RETURN count(path) as total")
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
    res = connector.query("MATCH path=(e)<-[inter:INTERMEDIARY_OF]-(i)-[off:OFFICER_OF]->(e) where off.toDelete_GT=True RETURN count(path) as total")
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
    res = connector.query("MATCH path=(e)<-[inter:INTERMEDIARY_OF]-(i)-[off:OFFICER_OF]->(e) where inter.toDelete_GT=True RETURN count(path) as total")
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
    res = connector.query("MATCH path=(o:Officer)-[of:OFFICER_OF]->(e:Entity) WHERE o.networth < 17000000 and of.toDelete_GT=True RETURN count(path) as total")
    num = res[0]['total']
    pd_tp+=0.5*num
    pd_fp+=0.5*num
    pd_fn+=0.5*num
    pu_tp+=0
    pu_fp+=num
    pu_fn+=0
    ps_tp+=0.66*num
    ps_fp+=0.33*num
    ps_fn+=0.33*num
    
    
    
    

    res = connector.query("MATCH path=(o:Officer)-[of:OFFICER_OF]->(e:Entity) WHERE o.networth_GT < 17000000 and of.toDelete=False  RETURN count(path) as total")
    num = res[0]['total']
    print(num)
    pd_tp+=0
    pd_fp+=0
    pd_fn+=num
    pu_tp+=0.5*num
    pu_fp+=0.5*num
    pu_fn+=0
    pu_tp+=0.5*num
    pu_fp+=0.5*num
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