set -e
#set -x
timestamp() {
	echo "[$(date +'%Y-%m-%d %H:%M:%S')]"
}

# No commit before this will work
FIRST_COMMIT="${FIRST_COMMIT:-211dec4d65d33da0977cbb45283d7d928701c947}"
REPO=https://github.com/bertmaher/pytorch.git
BRANCH=pytorch_fusion
CLONE_DIR="${CLONE_DIR:-nnc_bench}"
ENV_DIR="${ENV_DIR:-nnc_bench_env}"
ENV_DIR=$(realpath $ENV_DIR)
SQLITE_DB="${SQLITE_DB:-benchmarks.db}"
SQLITE_DB=$(realpath $SQLITE_DB)
LOG_FILE="${LOG_FILE:-bench.log}"
LOG_FILE=$(realpath $LOG_FILE)
sqlite3 $SQLITE_DB "create table IF NOT EXISTS benchmarks (commit_hash TEXT PRIMARY KEY, data TEXT);"
rm -f $LOG_FILE

# setup python environment
[ ! -d $ENV_DIR ] && python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate

# Grab the latest code or whatever is cloned into the folder
[ ! -d $CLONE_DIR ] && git clone $REPO --recursive $CLONE_DIR
pushd $CLONE_DIR

git checkout $BRANCH 
commits=($(git rev-list $FIRST_COMMIT..HEAD))
echo "Benchmarking ${#commits[@]} commits"
for ((i=${#commits[@]}-1; i>=0; i--)); do
	commit="${commits[$i]}"
    res=$(sqlite3 $SQLITE_DB "select exists(select 1 from benchmarks where commit_hash=\"$commit\")")
    if [[ $res == "1" ]]; then
        echo "Commit $commit already benchmarked, skipping"
        continue
    fi
	date=$(git show -s --format=%ci $commit)
	echo "$(timestamp) Building commit $commit from $date"
	git checkout $commit &>> $LOG_FILE
	pip install -r requirements.txt &>> $LOG_FILE
    # Try building, if it doesn't work just skip
    CC=gcc CXX=g++ USE_LLVM=/usr/lib/llvm-9/ USE_MKL=1 BUILD_CAFFE2_OPS=0 USE_CUDA=1 USE_CUDNN=1 python setup.py install &>> $LOG_FILE || continue
	echo "$(timestamp) Running commit $commit from $date"
	pushd benchmarks &>> $LOG_FILE
	data=$(python -m tensorexpr mul --device gpu,cpu --mode fwd --jit_mode trace --cuda_fuser=te --output=json || echo "error")
    if [ "$data" = "error" ]; then
        continue
    fi
	popd &>> $LOG_FILE
	echo $data
	sqlite3 $SQLITE_DB "replace into benchmarks (commit_hash, data) values ('$commit', '$data');"
done
popd
