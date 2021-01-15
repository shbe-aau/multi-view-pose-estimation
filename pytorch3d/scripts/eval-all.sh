#Eval all objs with union loss from paper
for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
    bash run-eval.sh ${OBJ_ID} "all-objs-vsd-union-99epochs" "pytorch3d/output/depth/all-objs-vsd-union/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
    wait
    bash run-eval.sh ${OBJ_ID} "all-objs-vsd-union-99epochs" "pytorch3d/output/depth/all-objs-vsd-union/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
    wait
    #bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "train" "tless"
    #wait
    #bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "test" "tless"
    #wait
done

# #Eval 2nd run of all objects with MSPD and MSSD
# for OBJ_ID in 25 #10 17 19
# do
#     bash run-eval.sh ${OBJ_ID} "add-pose-99epochs" "pytorch3d/output/depth/obj25-add-pose/models/model-epoch99.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "add-pose-99epochs" "pytorch3d/output/depth/obj25-add-pose/models/model-epoch99.pt" "test" "tless"
#     wait
# done

# #Eval 2nd run of all objects with MSPD and MSSD
# for OBJ_ID in 01 02 #03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "all-objs-2nd-run-metrics-99epochs" "pytorch3d/output/depth/all-objs-2nd-run/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "all-objs-2nd-run-metrics-99epochs" "pytorch3d/output/depth/all-objs-2nd-run/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "test" "tless"
#     wait
# done

# #Eval vsd intersection vs union
# for OBJ_ID in 10 17 19
# do
#     bash run-eval.sh ${OBJ_ID} "union-max20-60epochs" "pytorch3d/output/depth/loss-test-paper/union-max20/obj${OBJ_ID}/models/model-epoch60.pt" "test" "tless"
#     bash run-eval.sh ${OBJ_ID} "intersection-max20-60epochs" "pytorch3d/output/depth/loss-test-paper/intersection-max20/obj${OBJ_ID}/models/model-epoch60.pt" "test" "tless"
#     bash run-eval.sh ${OBJ_ID} "paper-vsd-log-60epochs" "pytorch3d/output/depth/loss-test-paper/paper-vsd-log/obj${OBJ_ID}/models/model-epoch60.pt" "test" "tless"
#     bash run-eval.sh ${OBJ_ID} "paper-vsd-max20-60epochs" "pytorch3d/output/depth/loss-test-paper/paper-vsd-max20/obj${OBJ_ID}/models/model-epoch60.pt" "test" "tless"
#     bash run-eval.sh ${OBJ_ID} "all-objs-2nd-run-60epochs" "pytorch3d/output/depth/all-objs-2nd-run/obj${OBJ_ID}/models/model-epoch60.pt" "test" "tless"
# done

# #Eval linemod objects
# for OBJ_ID in 05 # 03 05 10 #14 15
# do
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "train" "lm"
#     wait
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "test" "lm"
#     wait
#     bash run-eval.sh ${OBJ_ID} "nn-130epochs" "pytorch3d/output/depth/linemod/obj${OBJ_ID}/models/model-epoch130.pt" "train" "lm"
#     wait
#     bash run-eval.sh ${OBJ_ID} "nn-130epochs" "pytorch3d/output/depth/linemod/obj${OBJ_ID}/models/model-epoch130.pt" "test" "lm"
#     wait
#     bash run-eval.sh ${OBJ_ID} "l2-pose-99epochs" "pytorch3d/output/pose/linemod/obj${OBJ_ID}/models/model-epoch99.pt" "train" "lm"
#     wait
#     bash run-eval.sh ${OBJ_ID} "l2-pose-99epochs" "pytorch3d/output/pose/linemod/obj${OBJ_ID}/models/model-epoch99.pt" "test" "lm"
#     wait
# done

# #Eval run of all objects using a small model
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "all-objs-small-99epochs" "pytorch3d/output/depth/all-objs-small/obj${OBJ_ID}/models/model-epoch99.pt" "train"
#     wait
#     bash run-eval.sh ${OBJ_ID} "all-objs-small-99epochs" "pytorch3d/output/depth/all-objs-small/obj${OBJ_ID}/models/model-epoch99.pt" "test"
#     wait
# done

# #Eval 2nd run of all objects
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "all-objs-2nd-run-99epochs" "pytorch3d/output/depth/all-objs-2nd-run/obj${OBJ_ID}/models/model-epoch99.pt" "train"
#     wait
#     bash run-eval.sh ${OBJ_ID} "all-objs-2nd-run-99epochs" "pytorch3d/output/depth/all-objs-2nd-run/obj${OBJ_ID}/models/model-epoch99.pt" "test"
#     wait
# done

# # Eval L2 pose loss
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "l2-pose-100epochs" "pytorch3d/output/pose/all-objs/l2-pose/obj${OBJ_ID}/models/model-epoch100.pt" "train"
#     wait
#     bash run-eval.sh ${OBJ_ID} "l2-pose-100epochs" "pytorch3d/output/pose/all-objs/l2-pose/obj${OBJ_ID}/models/model-epoch100.pt" "test"
#     wait
# done


# # Eval trace pose loss
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "trace-pose-100epochs" "pytorch3d/output/pose/all-objs/trace-pose/obj${OBJ_ID}/models/model-epoch100.pt" "train"
#     wait
#     bash run-eval.sh ${OBJ_ID} "trace-pose-100epochs" "pytorch3d/output/pose/all-objs/trace-pose/obj${OBJ_ID}/models/model-epoch100.pt" "test"
#     wait
# done
