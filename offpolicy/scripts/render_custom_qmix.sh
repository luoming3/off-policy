#!/bin/sh
env="custom"
scenario="separated"
num_landmarks=3
num_agents=4
algo="mqmix"
exp="luoming"
seed_max=1
model_dir=""  # TODO: path to your model dir

current_dir=$(cd $(dirname $0); pwd)

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python -u ${current_dir}/render/render_custom.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --episode_length 1024 --batch_size 32 \
    --tau 0.005 --lr 7e-4 --hard_update_interval_episode 100 --num_env_steps 100000000 --n_rollout_threads 1 \
    --buffer_size 5000 --save_interval 50000 --model_dir ${model_dir} --render_episodes 1 --use_reward_normalization --use_wandb \
    --cuda --share_policy --render_env
    echo "training is done!"
done
