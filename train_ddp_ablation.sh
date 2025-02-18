set -x

export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IPV4_ONLY=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export NCCL_IB_GID_INDEX=3
ulimit -n 10240000
echo "Network settings are configured."


hostname=$1
num_machines=$3
num_processes=$((8 * num_machines))

IP=$(hostname -I | awk '{print $1}')
echo $IP

PORT=20687
NPROC_PER_NODE=8
NNODES=$num_machines

WORK_DIR=/nfs-26/maxiaoxiao/VAR_new

# ab_t5:
CONFIG_FILE=$WORK_DIR/configs/config_d16_ab_cross_t5.json
log_dir=/nfs-142/maxiaoxiao/workspace/var_rope_d16_theta_ab_cross_t5

# ab_rope:
CONFIG_FILE=$WORK_DIR/configs/config_d16_ab_rope.json
log_dir=/nfs-142/maxiaoxiao/workspace/var_rope_d16_theta_ab_rope

# ab_cross:
CONFIG_FILE=$WORK_DIR/configs/config_d16_ab_cross_256.json
log_dir=/nfs-142/maxiaoxiao/workspace/var_rope_d16_theta_ab_cross

nslookup $hostname

cd $WORK_DIR
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'


mkdir $log_dir
chmod -R 777 $log_dir
chmod -R 777 /nfs-142/maxiaoxiao/workspace

echo "Initializing environment on $($hostname)"
/opt/miniconda3/bin/python -m pip install lmdb tensorboard pytz timm opencv-python accelerate transformers==4.44.2 ftfy bs4
/opt/miniconda3/bin/python -m pip install -r $WORK_DIR/requirements.txt


/opt/miniconda3/bin/python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=0 --master_addr=$hostname --master_port=$PORT train.py \
--depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --config=$CONFIG_FILE