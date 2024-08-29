# Mix-Spectrum for Generalization in Visual Reinforcement Learning



## Abstract
Visual Reinforcement Learning (RL) trains agents on policies using raw images, showing potential for real-world applications. However, the limited diversity in training environments often results in overfitting, with agents underperforming in unseen environments due to a notable domain gap. To address challenges in traditional computer vision, image-based augmentation is used to increase data diversity, but its effectiveness is limited in RL tasks due to the intricate relationship between environmental dynamics and visual appearances. We introduce ``Mix-Spectrum,'' a novel frequency-based image augmentation method to enhance the agent's focus on semantic information and improve robustness in unseen environments. The proposed method transforms the image to the Fourier domain and mixes only the amplitudes with the other datasets. This allows for the augmentation of multiple datasets while preserving semantic information for robust generalization in visual reinforcement learning tasks. Through extensive experiments on the DMControl Generalization Benchmark (DMControl-GB), the paper demonstrates that Mix-Spectrum shows superior performance compared to existing methods in zero-shot generalization.

## Install MuJoCo
Download the MuJoCo version 2.0 binaries for Linux or OSX.


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

MUJOCO_GL='egl' python3 src/train.py  --domain_name walker --task_name walk --algorithm sac --seed 0 



