for dataset in faers; do
    # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
    for metric in difficulty; do
        for theta in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
            for budget in 80 240 480 540 720 960 1200 1440 1680 1920 2160 2403; do
                echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty budget: $budget"

                python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

                #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

                python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --save-assign-every 100

                python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
                
                done
            done 
    
    done
    #  for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
    #         for budget in 0 240 480 720 960 1200 1440 1680 1920 2160 2403; do
        
    #         python3 05_aggregate_quality.py --dataset "$dataset" --theta "$theta" --budget "$budget" 
    #         done
    #     done
        python3 07_quality.py --dataset "$dataset" 
    done

# for dataset in faers; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#         for metric in difficulty; do
#         for theta in 1; do
#             for budget in 240 480 720 960 1200 1440 1680 1920 2160 2403; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"

#                 python3 09_ratio.py --dataset "$dataset" --lam 0.01  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 100 --step0 1.0 --save-assign-every 200
#                 done
#             done 
#     done
#     done






for dataset in finbench; do
    # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
    for metric in difficulty; do
        for theta in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
            for budget in 0 270 900 1800 2700 3600 4500 6300 7200 8100 9000; do
                echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty budget: $budget"

                python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

                #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

                python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --save-assign-every 100

                python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
                
                done
            done 
    
    done
    #  for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
    #         for budget in 0 240 480 720 960 1200 1440 1680 1920 2160 2403; do
        
    #         python3 05_aggregate_quality.py --dataset "$dataset" --theta "$theta" --budget "$budget" 
    #         done
    #     done
        python3 07_quality.py --dataset "$dataset" 
    done


# for dataset in icij; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#         for theta in 0.5 0.6 0.7 0.8 0.9 1; do
#             for budget in 0 2108  3532 7064 10569 14128 17660 21192 24724 28256 3178835320 38861; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty budget: $budget"

#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --save-assign-every 100

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
                
#                 done
#             done 
    
#     done
#     #  for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#     #         for budget in 0 240 480 720 960 1200 1440 1680 1920 2160 2403; do
        
#     #         python3 05_aggregate_quality.py --dataset "$dataset" --theta "$theta" --budget "$budget" 
#     #         done
#     #     done
#         python3 07_quality.py --dataset "$dataset" 
#     done

# for dataset in snb; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#         for theta in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#             for budget in 0	2620 6549 13098 19647 26196 32745 39294 45843 52392 58941 65493 72044; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty budget: $budget"

#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 10 --step0 1.0 --save-assign-every 100

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
                
#                 done
#             done 
    
#     done
    #  for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
    #         for budget in 0 240 480 720 960 1200 1440 1680 1920 2160 2403; do
        
    #         python3 05_aggregate_quality.py --dataset "$dataset" --theta "$theta" --budget "$budget" 
    #         done
    #     done
        python3 07_quality.py --dataset "$dataset" 
    done


# for dataset in faers; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#         for theta in 0.2 0.3 0.4 0.5 0.6; do
#             for budget in 20 40 60 80 100 120 140 160 180 200 220 240; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --save-assign-every 5

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done
#         for theta in 0.7; do
#             for budget in 20 40 60 80 100 120 140 160 180 200 220 240 480 540 600 660 720; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done
#         for theta in 0.8; do
#             for budget in 20 40 60 80 100 120 140 160 180 200 220 240 720 760 800 840 880 920 960; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done
#         for theta in 0.9; do
#             for budget in 20 40 60 80 100 120 140 160 180 200 220 240 960 1000 1040 1080 1120 1160 1200; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done
#         for theta in 1; do
#             for budget in 20 40 60 80 100 120 140 160 180 200 220 240 1440 1480 1520 1560 1600 1640 1680; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done
#     done 
# done


# for dataset in faers; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#         for theta in 0.2 0.3 0.4 0.5 0.6; do
#             for budget in 20 40 60 80 100 120 140 160 180 200 220 240; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 write_res.py --dataset "$dataset" --metric "$metric" --theta "$theta" --budget "$budget"
#                 done
#             done
#         for theta in 0.7; do
#             for budget in 20 40 60 80 100 120 140 160 180 200 220 240 480 540 600 660 720; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 write_res.py --dataset "$dataset" --metric "$metric" --theta "$theta" --budget "$budget"
#                 done
#             done
#         for theta in 0.8; do
#             for budget in 20 40 60 80 100 120 140 160 180 200 220 240 720 760 800 840 880 920 960; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 write_res.py --dataset "$dataset" --metric "$metric" --theta "$theta" --budget "$budget"
#                 done
#             done
#         for theta in 0.9; do
#             for budget in 20 40 60 80 100 120 140 160 180 200 220 240 960 1000 1040 1080 1120 1160 1200; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 write_res.py --dataset "$dataset" --metric "$metric" --theta "$theta" --budget "$budget"
#                 done
#             done
#         for theta in 1; do
#             for budget in 20 40 60 80 100 120 140 160 180 200 220 240 1440 1480 1520 1560 1600 1640 1680; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 write_res.py --dataset "$dataset" --metric "$metric" --theta "$theta" --budget "$budget"
#                 done
#             done
#     done 
# done


# for dataset in finbench; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#     for theta in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#             for budget in 900; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done
        # for theta in 0.3 0.4 0.5; do
        #     for budget in 90 180 270 360 450 540 630 720 810 900 990 1080 1170 1260 1350 1440 1530 1620 1710 1800; do
        #         echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


        #         python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

        #         #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

        #         python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --save-assign-every 5

        #         python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
        #         done
        #     done
        # for theta in 0.6 0.7; do
        #     for budget in 1890 1980 2070 2160 2250 2340 2430 2520 2610 2700; do
        #         echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


        #         python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

        #         #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

        #         python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

        #         python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
        #         done
        #     done
        # for theta in 0.7 0.8; do
        #     for budget in 2790 2880 2970 3060 3150 3240 3330 3420 3510 3600; do
        #         echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


        #         python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

        #         #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

        #         python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

        #         python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
        #         done
        #     done
        # for theta in 0.8 0.9; do
        #     for budget in 3690 3780 3870 3960 4050 4140 4230 4320 4410 4500; do
        #         echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


        #         python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

        #         #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

        #         python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

        #         python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
        #         done
        #     done
        # for theta in 0.9; do
        #     for budget in 4590 4680 4770 4860 4950 5040 5130 5220 5310 5400; do
        #         echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


        #         python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

        #         #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

        #         python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

        #         python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
        #         done
        #     done
        # for theta in 1; do
        #     for budget in 5490 5580 5670 5760 5850 5940 6030 6120 6210 6300; do
        #         echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


        #         python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

        #         #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

        #         python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

        #         python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
        #         done
        #     done
#     done 
# done

# for dataset in icij; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#     for theta in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#             for budget in 35320; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done
#         done
#     done



# for dataset in snb; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#     for theta in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#             for budget in 65493; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"


#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done
#         done
#     done


# for dataset in icij; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#         for theta in 0.5 0.6 0.7; do
#             for budget in 353 706 1059 1413 1765 2108 2451 2784 3137 3633; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"

#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"

#                 python3 write_res.py --dataset "$dataset" --metric "$metric" --theta "$theta" --budget "$budget"
#                 done
#             done
        
#     done 
# done



# for dataset in snb; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#         for theta in 0.2 0.3; do
#             for budget in 655 1310 1965 2620 3275 3930 4585 5240 5895 6500; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"

#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"

#                 python3 write_res.py --dataset "$dataset" --metric "$metric" --theta "$theta" --budget "$budget"
#                 done
#             done
        
#     done 
# done


# for dataset in finbench; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#         for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#             for budget in 0 810 1636 2454 3272 4090 4908 5726 6544 8180 9000; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: $metric"



#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 5

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done 
#         done 
#     for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#             for budget in 0 818 1636 2454 3272 4090 4908 5726 6544 8180 9000; do
        
#             python3 05_aggregate_quality.py --dataset "$dataset" --theta "$theta" --budget "$budget" 
#             done
#         done
#     done

# for dataset in finbench; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#         for metric in difficulty; do
#         for theta in 1; do
#             for budget in 810 1636 2454 3272 4090 4908 5726 6544 7200 8180 9000; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"

#                 python3 09_ratio.py --dataset "$dataset" --lam 0.01  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 50 --step0 1.0 --save-assign-every 200
#                 done
#             done 
#     done
#     done


# for dataset in icij; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#         for metric in difficulty; do
#         for theta in 1; do
#             for budget in 3532 7064 10569 14128 17660 21192 24724 28256 31788 38861; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"

#                 python3 09_ratio.py --dataset "$dataset" --lam 0.01  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --save-assign-every 200
#                 done
#             done 
#     done
#     done


# for dataset in snb; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#         for metric in difficulty; do
#         for theta in 1; do
#             for budget in 6549 13098 19647 26196 32745 39294 45843 52392 58941 72044; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: difficulty"

#                 python3 09_ratio.py --dataset "$dataset" --lam 0.01  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 10 --step0 1.0 --save-assign-every 200
#                 done
#             done 
#     done
#     done

# for dataset in icij; do
#     for metric in difficulty; do
#         for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#             for budget in 0 3532 7064 10569 14128 17660 21192 24724 28256 31788 38861; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: $metric"

#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 10 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done 
#         done 
#     done
# for dataset in faers; do
#     for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#         for theta in 1; do
#             for budget in 2403; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: $metric"

#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 10 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done 
#         done 
#     done 



# for dataset in finbench; do
#     for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#         for theta in 1; do
#             for budget in 9000; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: $metric"

#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 10 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done 
#         done 
#     done



# for dataset in snb; do
#     for metric in difficulty; do
#         for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#             for budget in 0	6549 13098 19647 26196 32745 39294 45843 52392 58941 72044; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: $metric"

#                 python3 01_precheck_mss.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 #python3 02_inner_exact.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 20 --step0 1.0 --cap-mults --save-assign-every 5

#                 python3 03_outer_pairfree.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 10 --step0 1.0 --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done 
#         done 
#     done

# for dataset in icij; do
#     # for metric in difficulty normalized_cs_cl normalized_pagerank normalized_degree; do
#     for metric in difficulty; do
#         for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#             for budget in 0 818 1636 2454 3272 4090 4908 5726 6544 8180 9000; do
#                 echo "Running precheck with dataset: $dataset, theta: $theta, metric: $metric"



#                 python3 01_precheck_ms.py --dataset "$dataset" --theta "$theta" --metric "$metric"  

#                 python3 02_inner_exact_ms.py --dataset "$dataset" --lam 0.0  --budget "$budget" --metric "$metric"  --theta "$theta" --iters 10 --step0 1.0 --cap-mults --save-assign-every 2

#                 python3 04_compute_quality.py --dataset "$dataset" --theta "$theta" --metric "$metric" --budget "$budget"
#                 done
#             done 
#         done 
#     # for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#     #         for budget in 0 818 1636 2454 3272 4090 4908 5726 6544 8180 9000; do
        
#     #         python3 05_aggregate_quality.py --dataset "$dataset" --theta "$theta" --budget "$budget" 
#     #         done
#     #     done
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





