# Arguments
OBJ_ID=$1 #e.g 10
APPROACH_NAME=$2 #e.g "sundermeyer"
TRAINED_MODEL=$3 #e.g "pytorch3d/output/pose/obj19-10-poses-aug-bg-dataset/models/model-epoch20.pt"
DATA_SPLIT=$4 #e.g. "train"

# Parameters
SHARED_FOLDER=$(realpath ~/share-to-docker)

# Docker commands
GENERAL_DOCKER="docker run --rm --runtime=nvidia --user=$( id -u $USER ):$( id -g $USER ) --volume=/etc/group:/etc/group:ro --volume=/etc/passwd:/etc/passwd:ro --volume=/etc/shadow:/etc/shadow:ro --volume=/etc/sudoers.d:/etc/sudoers.d:ro -v ${SHARED_FOLDER}:/shared-folder -w /shared-folder/AugmentedAutoencoder -e DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw"
MERGED_DOCKER="${GENERAL_DOCKER} --env PYTHONPATH=/shared-folder/AugmentedAutoencoder/bop_toolkit:$PYTHONPATH aae-pytorch3d"
AAE_DOCKER="${GENERAL_DOCKER} tensorflow-opencv-opengl"

# ----------------------------------------------------------
echo "Convert BOP dataset to pickle for easy processing"
BOP_PICKLE_PATH=${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/tless-${DATA_SPLIT}-obj${OBJ_ID}.p
if test -f "$BOP_PICKLE_PATH"; then
    echo " - BOP dataset in pickle format already exists, skipping..."
else
    if test "$DATA_SPLIT" = "test"; then
	${MERGED_DOCKER} python pytorch3d/utils/bop2pickle.py --datasets_path /shared-folder/bop/bop-tless-dataset/ --obj_ids ${OBJ_ID} --dataset_split ${DATA_SPLIT} #> log.out
    else
	${MERGED_DOCKER} python pytorch3d/utils/bop2pickle.py --datasets_path /shared-folder/bop/bop-tless-dataset/ --obj_ids ${OBJ_ID} --dataset_split ${DATA_SPLIT} > log.out
    fi
    wait
    echo "Move pickle to BOP dataset folder"
    mkdir -p ${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}
    mv tless-${DATA_SPLIT}-obj${OBJ_ID}.p ${BOP_PICKLE_PATH}
    wait
fi

echo "Eval method: ${APPROACH_NAME} on the pickled data"
if test "$APPROACH_NAME" = "sundermeyer"; then
    SM_PICKLE_PATH=${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/${APPROACH_NAME}-result-${DATA_SPLIT}-obj${OBJ_ID}.p
    if test -f "$SM_PICKLE_PATH"; then
	echo " - pickle with results from Sundermeyers approach already exists, skipping..."
    else
	cp ${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/tless-${DATA_SPLIT}-obj${OBJ_ID}.p ${SM_PICKLE_PATH}
	wait
	${AAE_DOCKER} python -m auto_pose.ae.images2poses cad_autoencoder/obj${OBJ_ID} /shared-folder/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/sundermeyer-result-${DATA_SPLIT}-obj${OBJ_ID}.p > log.out
	wait
	echo "Convert results from Sundermeyer to CSV for BOP toolkit"
	SM_CSV_PATH=${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_tless-${DATA_SPLIT}-primesense.csv
	if test -f "$SM_CSV_PATH"; then
	    echo " - CSV with results from Sundermeyers approach already exists, skipping..."
	else
	    ${MERGED_DOCKER} python pytorch3d/utils/eval-pickle.py -pi /shared-folder/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/${APPROACH_NAME}-result-${DATA_SPLIT}-obj${OBJ_ID}.p -o /shared-folder/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/sundermeyer-obj${OBJ_ID}_tless-${DATA_SPLIT}-primesense.csv > log.out #-op pytorch3d/data/t-less-obj${OBJ_ID}/cad/obj_${OBJ_ID}.ply
	    wait
	fi
    fi
else
    OUR_CSV_PATH=${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_tless-${DATA_SPLIT}-primesense.csv
    if test -f "$OUR_CSV_PATH"; then
	echo " - CSV with results already exists, skipping..."
    else
	ENCODER_WEIGHTS="pytorch3d/data/t-less-obj${OBJ_ID}/obj${OBJ_ID}-encoder.npy"
	${MERGED_DOCKER} python pytorch3d/utils/eval-pickle.py -pi /shared-folder/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/tless-${DATA_SPLIT}-obj${OBJ_ID}.p -o /shared-folder/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_tless-${DATA_SPLIT}-primesense.csv -ep ${ENCODER_WEIGHTS} -mp ${TRAINED_MODEL} > log.out #-op pytorch3d/data/t-less-obj${OBJ_ID}/cad/obj_${OBJ_ID}.ply
	wait
    fi

fi

echo "Run BOP eval"
RESULT_CSV=/shared-folder/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_tless-${DATA_SPLIT}-primesense.csv
BOP_EVAL_PATH=/shared-folder/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/eval
LOCAL_BOP_EVAL_PATH=${SHARED_FOLDER}/bop/bop-tless-dataset/tless/pickles/${DATA_SPLIT}/obj${OBJ_ID}/eval/${APPROACH_NAME}-obj${OBJ_ID}_tless-${DATA_SPLIT}-primesense
if test -d "$LOCAL_BOP_EVAL_PATH"; then
    echo " - eval dir from BOP already exists, skipping..."
else
    #if test "$DATA_SPLIT" = "train"; then
    TARGETS_FILE="/shared-folder/bop/bop-tless-dataset/tless/targets/${DATA_SPLIT}_targets_obj${OBJ_ID}.json"
    #else
	#TARGETS_FILE="/shared-folder/bop/bop-tless-dataset/tless/targets/test_targets_bop19.json"
    #fi
    ${MERGED_DOCKER} python bop_toolkit/scripts/eval_bop19.py --result_filenames=${RESULT_CSV} --targets_filename=${TARGETS_FILE} --datasets_path /shared-folder/bop/bop-tless-dataset/ --eval_path ${BOP_EVAL_PATH}
fi

echo "Show BOP performance"
${MERGED_DOCKER} python bop_toolkit/scripts/show_performance_bop19.py --result_filenames=${RESULT_CSV} --datasets_path /shared-folder/bop/bop-tless-dataset/ --eval_path ${BOP_EVAL_PATH}
