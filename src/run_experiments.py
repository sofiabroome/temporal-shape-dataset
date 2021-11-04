import os
import json

CONFIG_PATH = 'configs/transfconv.json'
MEMORY = 64 #GB
GPU_NO = 4
CPUS_PER_TASK = 4

# dim_head = dim/head
dims = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48]

for dim in dims:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        config['dim'] = dim
        json.dump(config, open(CONFIG_PATH, "w"), indent = 4)
        job_id = f'5dot_8head_{dim}dim'
        os.system(f'srun --gres=gpu:4{GPU_NO}--mem={MEMORY}GB --cpus-per-task={CPUS_PER_TASK} python main.py --config {CONFIG_PATH} --job_identiier {job_id} --log_every_n_steps=5 --gpus={GPU_NO}')
