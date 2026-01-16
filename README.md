# GNN-BASED Interactive Graph Repairs




## 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies (suggested to use a virtual environment).

```bash
pip3 install -r requirements.txt
```

## Dataset

Load the following datasets in Neo4j (version used : community edition 5.12)

- [Star Wars](https://github.com/neo4j-graph-examples/star-wars)
- [FAERS](https://github.com/neo4j-graph-examples/healthcare-analytics)
- [LDBC FinBench](https://github.com/ldbc/ldbc_finbench_datagen) : Scaling factor 0.1   
- [ICIJ](https://github.com/neo4j-graph-examples/icij-paradise-papers) 
- [LDBC SNB](https://github.com/ldbc/ldbc_snb_datagen_spark) : Scaling factor 0.1 

For each dataset, copy the Cypher script located inside the /data folder (e.g. icij.cypher for ICIJ) and execute it in Neo4j.
Then, download the dataset with violations : 

```sql
CALL apoc.export.csv.all("<DATASET-NAME>_with_violations.csv", {})
```

## GNN Annotation

```bash
python3 testTensor.py <DATASET-FOLDER-NAME> <DATASET-CSV-FILE-NAME>

python3 train_heterogat.py <DATASET-FOLDER-NAME> <DATASET-CSV-FILE-NAME>

python3 annotate_graph.py <DATASET-FOLDER-NAME> <DATASET-CSV-FILE-NAME>

```

### EXPERIMENTS SETUP


```bash
python3 generate_grdgs.py --dataset <DATASET-FOLDER-NAME>
 
python3 generate_users.py --dataset <DATASET-FOLDER-NAME>

python3 00_prep_users.py --dataset <DATASET-FOLDER-NAME>
```

### EXPERIMENTS


```bash

./00_run_lagrangian.sh

./1_run_thetabudget.sh

./5_aggregate_quality.sh

python3 quality.py --dataset <DATASET-FOLDER-NAME>

```

## License

[MIT](https://choosealicense.com/licenses/mit/)
