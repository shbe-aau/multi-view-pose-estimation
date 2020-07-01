# Arguments
OBJ_ID=$1 #e.g 10
APPROACH_NAME=$2 #e.g "sundermeyer"
TRAINED_MODEL=$3 #e.g "pytorch3d/output/pose/obj19-10-poses-aug-bg-dataset/models/model-epoch20.pt"
DATA_SPLIT="TRAIN - not implemented. Datasplit=train is hard-coded"

# Parameters
SHARED_FOLDER=$(realpath ~/share-to-docker)

# Docker commands
GENERAL_DOCKER="docker run --rm --runtime=nvidia --user=$( id -u $USER ):$( id -g $USER ) --volume=/etc/group:/etc/group:ro --volume=/etc/passwd:/etc/passwd:ro --volume=/etc/shadow:/etc/shadow:ro --volume=/etc/sudoers.d:/etc/sudoers.d:ro -v ${SHARED_FOLDER}:/shared-folder -w /shared-folder/AugmentedAutoencoder -e DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw"
MERGED_DOCKER="${GENERAL_DOCKER} --env PYTHONPATH=/shared-folder/AugmentedAutoencoder/bop_toolkit:$PYTHONPATH aae-pytorch3d"
AAE_DOCKER="${GENERAL_DOCKER} tensorflow-opencv-opengl"

# ----------------------------------------------------------
echo "Convert BOP dataset to pickle for easy processing"
BOP_PICKLE_PATH=${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/tless-train-obj${OBJ_ID}.p
if test -f "$BOP_PICKLE_PATH"; then
    echo " - BOP dataset in pickle format already exists, skipping..."
else
    ${MERGED_DOCKER} python bop_toolkit/scripts/bop2pickle.py --datasets_path /shared-folder/bop/bop-tless-dataset/ --obj_ids ${OBJ_ID} --dataset_split train > log.out
    wait
    echo "Move pickle to BOP dataset folder"
    mkdir -p ${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}
    mv tless-train-obj${OBJ_ID}.p ${BOP_PICKLE_PATH}
    wait
fi

echo "Eval method: ${APPROACH_NAME} on the pickled data"
if test "$APPROACH_NAME" = "sundermeyer"; then
    SM_PICKLE_PATH=${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/${APPROACH_NAME}-result-train-obj${OBJ_ID}.p
    if test -f "$SM_PICKLE_PATH"; then
	echo " - pickle with results from Sundermeyers approach already exists, skipping..."
    else
	cp ${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/tless-train-obj${OBJ_ID}.p ${SM_PICKLE_PATH}
	wait
	${AAE_DOCKER} python -m auto_pose.ae.images2poses cad_autoencoder/obj${OBJ_ID} /shared-folder/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/sundermeyer-result-train-obj${OBJ_ID}.p > log.out
	wait
	echo "Convert results from Sundermeyer to CSV for BOP toolkit"
	SM_CSV_PATH=${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_tless-train-primesense.csv
	if test -f "$SM_CSV_PATH"; then
	    echo " - CSV with results from Sundermeyers approach already exists, skipping..."
	else
	    ${MERGED_DOCKER} python pytorch3d/eval-pickle.py -pi /shared-folder/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/${APPROACH_NAME}-result-train-obj${OBJ_ID}.p -o /shared-folder/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/sundermeyer-obj${OBJ_ID}_tless-train-primesense.csv > log.out #-op pytorch3d/data/t-less-obj${OBJ_ID}/cad/obj_${OBJ_ID}.ply
	    wait
	fi
    fi
else
    OUR_CSV_PATH=${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_tless-train-primesense.csv
    if test -f "$OUR_CSV_PATH"; then
	echo " - CSV with results already exists, skipping..."
    else
	ENCODER_WEIGHTS="pytorch3d/data/t-less-obj${OBJ_ID}/obj${OBJ_ID}-encoder.npy"
	${MERGED_DOCKER} python pytorch3d/eval-pickle.py -pi /shared-folder/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/tless-train-obj${OBJ_ID}.p -o /shared-folder/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_tless-train-primesense.csv -ep ${ENCODER_WEIGHTS} -mp ${TRAINED_MODEL} > log.out #-op pytorch3d/data/t-less-obj${OBJ_ID}/cad/obj_${OBJ_ID}.ply
	wait
    fi

fi

echo "Run BOP eval"
RESULT_CSV=/shared-folder/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_tless-train-primesense.csv
BOP_EVAL_PATH=/shared-folder/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/eval
LOCAL_BOP_EVAL_PATH=${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/obj${OBJ_ID}/eval/${APPROACH_NAME}-obj${OBJ_ID}_tless-train-primesense
if test -d "$LOCAL_BOP_EVAL_PATH"; then
    echo " - eval dir from BOP already exists, skipping..."
else
    ${MERGED_DOCKER} python bop_toolkit/scripts/eval_bop19.py --result_filenames=${RESULT_CSV} --targets_filename=/shared-folder/bop/bop-tless-dataset/tless/targets/train_targets_obj${OBJ_ID}.json --datasets_path /shared-folder/bop/bop-tless-dataset/ --eval_path ${BOP_EVAL_PATH}
fi

echo "Show BOP performance"
${MERGED_DOCKER} python bop_toolkit/scripts/show_performance_bop19.py --result_filenames=${RESULT_CSV} --datasets_path /shared-folder/bop/bop-tless-dataset/ --eval_path ${BOP_EVAL_PATH}
