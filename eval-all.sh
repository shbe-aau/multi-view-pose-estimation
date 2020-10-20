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

# Eval object 1-20, depth only
for OBJ_ID in 10 #02 05 09 10 17 19 #11 15 17 19 20 #01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
do
   bash run-eval.sh ${OBJ_ID} "chamfer-fixed-view" "pytorch3d/output/depth/chamfer/obj${OBJ_ID}/models/model-epoch180.pt" "train"
   wait
done
