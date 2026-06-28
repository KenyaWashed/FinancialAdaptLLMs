DOMAIN=$1
MODEL=$2
ADD_BOS=$3
MODEL_PARALLEL=$4
N_GPU=$5
MODEL_TYPE=${6:-${MODEL_TYPE:-huggingface}}
VOCAB_SIZE=${7:-${VOCAB_SIZE:-50000}}
shift 6
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == vocab_size=* ]]; then
        VOCAB_SIZE="${arg#vocab_size=}"
    elif [[ "$arg" == model_type=* ]]; then
        MODEL_TYPE="${arg#model_type=}"
    else
        EXTRA_ARGS+=("$arg")
    fi
 done
if [ -z "$MODEL_TYPE" ]; then
    echo "ERROR: MODEL_TYPE is required as the 6th positional argument or model_type=... override."
    exit 1
fi
OUTPUT_DIR='/tmp/output/' # for saving the prediction files
RES_DIR='/tmp/res/' # for saving the evaluation scores of each task
CACHE_DIR='/tmp/cache' # for caching hf models and datasets

if [ ${DOMAIN} == 'biomedicine' ]; then
    TASK='MQP+PubMedQA+RCT+USMLE+ChemProt'
elif [ ${DOMAIN} == 'finance' ]; then
    TASK='NER+FPB+FiQA_SA+Headline+ConvFinQA'
elif [ ${DOMAIN} == 'law' ]; then
    TASK='CaseHOLD+SCOTUS+UNFAIR_ToS'
else
    TASK=${DOMAIN}
fi

echo "Domain-specific tasks: ${TASK}"
echo "MODEL: ${MODEL}"
echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "ADD_BOS: ${ADD_BOS}"
echo "MODEL_PARALLEL: ${MODEL_PARALLEL}"
echo "N_GPU: ${N_GPU}"

if [ ${MODEL_TYPE} == 'custom' ]; then
    echo "Custom model detected; proceeding. Ensure the custom model supports multi-GPU (DDP) when using N_GPU>1."
fi

if [ ${N_GPU} == '8' ]; then
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch  --num_processes ${N_GPU} --multi_gpu \
        inference.py task_name=${TASK} model_name=${MODEL} model_type=${MODEL_TYPE} add_bos_token=${ADD_BOS} vocab_size=${VOCAB_SIZE} ${EXTRA_ARGS[@]} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${MODEL_PARALLEL} \
        hydra.run.dir=/tmp
elif [ ${N_GPU} == '4' ]; then
    CUDA_VISIBLE_DEVICES='0,1,2,3' accelerate launch  --num_processes ${N_GPU} --multi_gpu \
        inference.py task_name=${TASK} model_name=${MODEL} model_type=${MODEL_TYPE} add_bos_token=${ADD_BOS} vocab_size=${VOCAB_SIZE} ${EXTRA_ARGS[@]} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${MODEL_PARALLEL} \
        hydra.run.dir=/tmp
elif [ ${N_GPU} == '2' ]; then
    CUDA_VISIBLE_DEVICES='0,1' accelerate launch  --num_processes ${N_GPU} --multi_gpu \
        inference.py task_name=${TASK} model_name=${MODEL} model_type=${MODEL_TYPE} add_bos_token=${ADD_BOS} vocab_size=${VOCAB_SIZE} ${EXTRA_ARGS[@]} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${MODEL_PARALLEL} \
        hydra.run.dir=/tmp
elif [ ${N_GPU} == '1' ]; then
    CUDA_VISIBLE_DEVICES='0' accelerate launch  --num_processes 1 \
        inference.py task_name=${TASK} model_name=${MODEL} model_type=${MODEL_TYPE} add_bos_token=${ADD_BOS} vocab_size=${VOCAB_SIZE} ${EXTRA_ARGS[@]} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${MODEL_PARALLEL} \
        hydra.run.dir=/tmp
fi