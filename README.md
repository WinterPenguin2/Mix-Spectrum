# Mix-Spectrum for Generalization in Visual Reinforcement Learning
Spectrum Random Erasing for Generalization in Image-based Reinforcement Learning



## Setup
We assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:

## Install MuJoCo
Download the MuJoCo version 2.1 binaries for Linux or OSX.

Extract the downloaded mujoco210 directory into \~/.mujoco/mujoco210.

If you want to specify a nonstandard location for the package, use the env variable MUJOCO_PY_MUJOCO_PATH.  
pip3 install -U 'mujoco-py<2.2,>=2.1'


## Install DMControl
conda env create -f setup/conda.yml

conda activate dmcgb

sh setup/install_envs.sh




## Usage
## DMControl Benchmark

from env.wrappers import make_env  
env = make_env(  
        domain_name=args.domain_name,  
        task_name=args.task_name,  
        seed=args.seed,  
        episode_length=args.episode_length,  
        action_repeat=args.action_repeat,  
        image_size=args.image_size,  
        mode='train'  
)
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)  


You can try other environments easily.



## Training

MUJOCO_GL='egl' CUDA_VISIBLE_DEVICES=10  python3 src/train.py   --algorithm drq_aug   --seed 0 --tag SRM  --augmentation random_mask_freq; 



