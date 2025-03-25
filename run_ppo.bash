python ppo.py --env_id="Plate-v1" --exp-name="state-plate" \
    --num_envs=32 --update_epochs=4 --num_minibatches=32 \
    --total_timesteps=800_0000 --eval_freq=25 --num-steps=20 \
    --track \
    # --evaluate --checkpoint="/home/ysq/project/maniskill/runs/state-plate/ckpt_177.pt"