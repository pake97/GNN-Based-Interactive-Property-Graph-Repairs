for dataset in faers finbench; do
    # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
    for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
        for theta in 1; do
            for budget in 100000000; do
                echo "Running precheck with dataset: $dataset, theta: $theta, metric: $metric"



                python3 01_precheck.py --dataset "$dataset" --theta "$theta" --metric "$metric" --concat 

                python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 1 --step0 1.0 --cap-mults --save-assign-every 1

                python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
                done
            done 
        done 
    for theta in 1.0; do
        for budget in 100000000; do
        
            python3 05_aggregate_quality.py --dataset "$dataset" --theta "$theta" --budget "$budget" 
            done
        done
    done


for dataset in faers finbench; do
    # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
    for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
        for theta in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
            for budget in 100000000; do
                echo "Running precheck with dataset: $dataset, theta: $theta, metric: $metric"



                python3 01_precheck.py --dataset "$dataset" --theta "$theta" --metric "$metric" --concat 

                python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 1 --step0 1.0 --cap-mults --save-assign-every 1

                python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
                done
            done 
        done 
    for theta in 1.0; do
        for budget in 100000000; do
        
            python3 05_aggregate_quality.py --dataset "$dataset" --theta "$theta" --budget "$budget" 
            done
        done
    done


# for dataset in icij; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#         for theta in 1.0; do
#             for budget in 100000000; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: $metric"



#                 python3 01_precheck_ms.py --dataset "$dataset" --theta "$theta" --metric "$metric" --concat 

#                 python3 02_inner_exact_ms.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 1 --step0 1.0 --cap-mults --save-assign-every 1

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done 
#         done 
#     for theta in 1.0; do
#         for budget in 100000000; do
        
#             python3 05_aggregate_quality.py --dataset "$dataset" --theta "$theta" --budget "$budget" 
#             done
#         done
#     done




# for dataset in snb; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#         for theta in 1.0; do
#             for budget in 100000000; do
#                 echo "--------------------------------------------------"
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: $metric"
#                 echo "--------------------------------------------------"
#                 echo "precheck_mss"
#                 echo "--------------------------------------------------"
#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric" 
#                 echo "--------------------------------------------------"
#                 echo "outer_pairfree"
#                 echo "--------------------------------------------------"
#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 1 --step0 1.0  --save-assign-every 1
#                 echo "--------------------------------------------------"
#                 echo "compute_quality"
#                 echo "--------------------------------------------------"
#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done 
#         done 
#     for theta in 1.0; do
#         for budget in 100000000; do
#             echo "--------------------------------------------------"
#             echo "aggregate_quality"
#             echo "--------------------------------------------------"
#             python3 05_aggregate_quality.py --dataset "$dataset" --theta "$theta" --budget "$budget" 
#             done
#         done
#     done


