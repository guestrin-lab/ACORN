export debugSearchFlag=0
#! /bin/bash



#cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B   build
# build with openmp
#cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release  -DCMAKE_C_FLAGS="-fopenmp" -DCMAKE_CXX_FLAGS="-fopenmp" -B   build
#make -C build -j faiss
#make -C build utils
make -C build -j acorn


#../DiskANN_CPP/DiskANN/build/apps/utils/compute_filtered_groundtruth  --K 10 --base_file ./base.fbin --base_labels ./base_labels.txt --dist_fn l2 --data_type float --gt_file gt_1M.bin --query_file query.fbin --query_label ./query_labels.txt 

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")


 # run  
N=1000000
nq=10 
gamma=1 # expansion factor
dataset=amazon
M_beta=16 # max degree, use same M and M_beta for now
gt="dense_gt_1M.bin.txt"
efc=100 # build limit
efs=0 # search limi0t, 0 for build only, >1 for search only

header="      N,   nq,   efc,  efs,   M,    M_beta, gamma,    recall,   query_time, bitmap_time" 
summary_file=${now}_experiments/summary_${dataset}.txt
rm -f $summary_file
echo $header>> $summary_file
echo $header
for M_beta in 32 64; do
    for gamma in 1 10; do
            parent_dir=${now}_experiments
            mkdir -p ${parent_dir}
            dir=${parent_dir}/MB${M_beta}_gamma${gamma}
            mkdir -p ${dir}
            efs=0
            TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${dataset}_n=${N}_nq=${nq}_efs=${efs}_gamma=${gamma}_M_beta=${M_beta}.txt
            #echo "Running acorn with N=${N}, nq=${nq}, efc=${efc} efs=${efs}, M=${M_beta}, M_beta=${M_beta}, gamma=${gamma}"
            #./build/demos/acorn $N $nq ../$dataset/ . . $gt $efc $M_beta $efs &>> ${dir}/summary_${dataset}_n=${N}_nq=${nq}_efs=${efs}_gamma=${gamma}_M_beta=${M_beta}.txt
            for efs in 40 80 150; do
                nq=3000
                txtfile=${dir}/summary_${dataset}_n=${N}_nq=${nq}_efs=${efs}_gamma=${gamma}_M_beta=${M_beta}.txt
                rm -f $txtfile
                TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${txtfile}
                ./build/demos/acorn $N $nq ../$dataset/ . . $gt $efc $M_beta $efs &>> ${txtfile}
                # Extract Query time
                query_time=$(grep -oP 'Done Query \K[0-9.]+(?=)' "$txtfile")
                bitmap_time=$(grep -oP 'Done filter_ids_map \K[0-9.]+(?=)' "$txtfile")
                # Extract Recall@10
                recall=$(grep -oP 'Recall@10: \K[0-9.]+(?=)' "$txtfile")
                 # Print the results
                #echo " N, nq,  efc,  efs,, M, M_beta, gamma, recall, query_time"
                row="$N,$nq,$efc,$efs,$M_beta,$M_beta,$gamma,$recall,$query_time, $bitmap_time"
                echo $row >> $summary_file
                echo $row
             done
    done
done







