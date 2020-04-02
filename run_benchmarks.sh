set -e

BRANCH="${BRANCH:-master}"
CPU="${CPU:-95}"
GPU="${GPU:-7}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

REPO=https://github.com/pytorch/pytorch.git
VISION_REPO=https://github.com/pytorch/vision.git

CLONE_DIR="${CLONE_DIR:-${SCRIPT_DIR}/nnc_bench_code}"
CLONE_DIR=$(realpath $CLONE_DIR)
mkdir -p $CLONE_DIR

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/nnc_bench_logs}"
LOG_DIR=$(realpath $LOG_DIR)
DATE=$(date +'%Y_%m_%d-%H_%M_%S')
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${DATE}.log}"
LOG_FILE=$(realpath $LOG_FILE)
rm -f $LOG_FILE

# setup python environment
ENV_DIR="${ENV_DIR:-${SCRIPT_DIR}/nnc_bench_env}"
ENV_DIR=$(realpath $ENV_DIR)
[ ! -d $ENV_DIR ] && python3 -m venv $ENV_DIR &>> $LOG_FILE
source $ENV_DIR/bin/activate

RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/nnc_bench_results}"
RESULT_DIR=$(realpath $RESULT_DIR)
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

# Grab the latest code or whatever is cloned into the folder
pushd $CLONE_DIR

[ ! -d pytorch ] && git clone $REPO --recursive &>> $LOG_FILE
[ ! -d vision ] && git clone $VISION_REPO --recursive &>> $LOG_FILE

pushd pytorch
git checkout master
git branch -D upstream &> /dev/null || true
git fetch origin ${BRANCH}:upstream
git checkout upstream
git submodule update --init --recursive
git submodule foreach --recursive git clean -xfd
git submodule foreach --recursive git reset --hard
commit=$(git rev-list HEAD~1..HEAD)
date=$(git show -s --format=%ci $commit)
unix_ts=$(date --date="$date" +'%s')
popd

pushd vision
git checkout master
git branch -D upstream &> /dev/null || true
git fetch origin master:upstream
git checkout upstream
git submodule update --init --recursive
popd

popd

RESULT_FILE="${RESULT_FILE:-${RESULT_DIR}/bench_${unix_ts}_${commit}.result}"
RESULT_FILE=$(realpath $RESULT_FILE)

if [ -s "$RESULT_FILE" ]; then
  echo "$RESULT_FILE" already exists, refusing to overwrite...
  exit 1
fi
touch $RESULT_FILE

pushd $CLONE_DIR
pushd pytorch

echo "Building commit $commit from $date" &>> $LOG_FILE
pip install -r requirements.txt &>> $LOG_FILE
CC=gcc CXX=g++ USE_LLVM=/usr/lib/llvm-9/ USE_MKL=1 BUILD_CAFFE2_OPS=0 USE_CUDA=1 USE_CUDNN=1 python setup.py install &>> $LOG_FILE
popd
pushd vision
python setup.py install &>> $LOG_FILE
popd
popd
pip install scipy
CUDA_VISIBLE_DEVICES=${GPU} numactl -C ${CPU} python ${SCRIPT_DIR}/benchmarks/models.py --cuda --tensorexpr --seconds 10 >> $RESULT_FILE
CUDA_VISIBLE_DEVICES=${GPU} numactl -C ${CPU} python ${SCRIPT_DIR}/benchmarks/models.py --cuda --seconds 10 >> $RESULT_FILE
numactl -C ${CPU} python ${SCRIPT_DIR}/benchmarks/models.py --tensorexpr >> $RESULT_FILE
numactl -C ${CPU} python ${SCRIPT_DIR}/benchmarks/models.py >> $RESULT_FILE
