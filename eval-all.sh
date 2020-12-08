#Eval object 1-20 on test split
for OBJ_ID in 02 05 08 09 10 12 17 19 #01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
do
    bash run-eval.sh ${OBJ_ID} "med-model-log-fixed-wolfram-70epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/med-model-log-fixed-wolfram/obj${OBJ_ID}/models/model-epoch70.pt" "test"
    wait
    bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "test"
    wait
done


## Eval object 17
#bash run-eval.sh 17 "test" "pytorch3d/output/depth/vsd-predicted-view-degrees/med-model-log-fixed-wolfram/obj17/models/model-epoch70.pt" "test"



# #Eval object 1-20
# for OBJ_ID in 02 05 08 09 10 12 17 19 #01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
# do
#     bash run-eval.sh ${OBJ_ID} "posemax40-6views-140epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/posemax40-6views/obj${OBJ_ID}/models/model-epoch140.pt" "train" -1
#     wait
# done

# # Eval object 17
# bash run-eval.sh 17 "med-model-log-fixed-wolfram-40epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/med-model-log-fixed-wolfram/obj17/models/model-epoch40.pt" "train" -1
# wait

# bash run-eval.sh 17 "posemax40-6views-40epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj17-param-sweep/posemax40-6views/obj17/models/model-epoch40.pt" "train" -1
# wait

# bash run-eval.sh 17 "fixed-wolfram-40epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax40-6views/obj17-fixed/models/model-epoch40.pt" "train" -1
# wait

# bash run-eval.sh 17 "fixed-wolfram-fixed-loss-40epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax40-6views/obj17-fixed-fixed/models/model-epoch40.pt" "train" -1
# wait


# # Eval object 10 param sweep
# bash run-eval.sh 10 "obj10-posemax20-6views-150epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax20-6views/obj10/models/model-epoch150.pt" "train" -1
# wait

# bash run-eval.sh 10 "obj10-posemax40-6views-150epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax40-6views/obj10/models/model-epoch150.pt" "train" -1
# wait

# bash run-eval.sh 10 "obj10-posemax10-12views-40epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax10-12views/obj10/models/model-epoch40.pt" "train" -1
# wait

# bash run-eval.sh 10 "obj10-posemax10-6views-40epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax10-6views/obj10/models/model-epoch40.pt" "train" -1
# wait

# bash run-eval.sh 10 "obj10-posemax10-8views-40epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax10-8views/obj10/models/model-epoch40.pt" "train" -1
# wait

# bash run-eval.sh 10 "obj10-posemax20-12views-40epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax20-12views/obj10/models/model-epoch40.pt" "train" -1
# wait

# bash run-eval.sh 10 "obj10-posemax20-6views-40epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax20-6views/obj10/models/model-epoch40.pt" "train" -1
# wait

# bash run-eval.sh 10 "obj10-posemax40-12views-40epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax40-12views/obj10/models/model-epoch40.pt" "train" -1
# wait

# # Eval object 17
# bash run-eval.sh 17 "test" "pytorch3d/output/depth/vsd-predicted-view-degrees/med-model-log-fixed-wolfram/obj17/models/model-epoch70.pt" "test" 16

# bash run-eval.sh 17 "test" "pytorch3d/output/depth/vsd-predicted-view-degrees/med-model-log-fixed-wolfram/obj17/models/model-epoch70.pt" "train" -1

# for OBJ_ID in 17 #09 10 #02 04 05 06 07 08 09 10 11 12 14 15 17 18 19 20
# do
#     for SCENE_ID in 15 16 19 #09 10 #02 04 05 06 07 08 09 10 11 12 14 15 17 18 19 20
#     do
# 	bash run-eval.sh ${OBJ_ID} "vsd-predicted-med-model-log-fixed-wolfram-70epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/med-model-log-fixed-wolfram/obj${OBJ_ID}/models/model-epoch70.pt" "test" ${SCENE_ID}
# 	wait
#     done
# done




# Eval object 19
#bash run-eval.sh 19 "sundermeyer" ""

# Eval object 10
#bash run-eval.sh 10 "sundermeyer" "" "train"

# Eval object 17
#bash run-eval.sh 17 "sundermeyer" ""

# Eval object 1-20, pose then depth
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
# do
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" ""
#     wait
#     bash run-eval.sh ${OBJ_ID} "pose-pretrain" "pytorch3d/output/pose-2-depth/obj1-20/obj${OBJ_ID}-random-multiview/models/model-epoch39.pt"
#     wait
#     bash run-eval.sh ${OBJ_ID} "pose-random-multiview-depth" "pytorch3d/output/pose-2-depth/obj1-20/obj${OBJ_ID}-random-multiview/models/model-epoch99.pt"
#     wait
# done

# wait

#bash run-eval.sh 10 "predicted-view-cubic-epoch100" "pytorch3d/output/depth/clamped-predicted-view/obj10-cubic/models/model-epoch99.pt" "train"
#wait

# # Eval object 1-20, depth only
# for OBJ_ID in 15 12 #19 #09 10 19 02 05 08 12 15 17 #02 05 09 10 12 17 19 #02 05 09 10 17 19 #11 15 17 19 20 #01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
# do
#     bash run-eval.sh ${OBJ_ID} "vsd-predicted-med-model-log-fixed-wolfram-70epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/med-model-log-fixed-wolfram/obj${OBJ_ID}/models/model-epoch70.pt" "test"
#     wait
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "test"
#     wait
# done


# # Eval object 1-20, depth only
# for OBJ_ID in 09 10 #02 04 05 06 07 08 09 10 11 12 14 15 17 18 19 20
# do
#     bash run-eval.sh ${OBJ_ID} "vsd-predicted-med-model-log-fixed-wolfram-50epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/med-model-log-fixed-wolfram/obj${OBJ_ID}/models/model-epoch50.pt" "train"
#     wait
# done



# # Eval object 1-20, depth only
# for LR in 1 2 3 4 5 6
# do
#     bash run-eval.sh 10 "obj10-posemax40-6views-lr${LR}-50epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax40-6views-lr${LR}/obj10/models/model-epoch50.pt" "train"
#     wait
# done


# # Eval object 10
# for EPOCH in 70 60 50 40 30 20 #10
# do
#     for POSEMAX in 20 40 80
#     do
# 	for VIEW in 4 6 8
# 	do
# 	    bash run-eval.sh 10  "obj10-posemax${POSEMAX}-${VIEW}views-${EPOCH}epochs" "pytorch3d/output/depth/vsd-predicted-view-degrees/obj10-param-sweep/posemax${POSEMAX}-${VIEW}views/obj10/models/model-epoch${EPOCH}.pt" "train"
# 	    wait
# 	done
#     done
# done
