<h2 style="text-align: center;">Invisibility Cloak: Proactive Defense Against Visual Game Cheating</h2>

<br>
<div align="center">
Chenxin Sun<sup>*</sup>, Kai Ye<sup>*</sup>, Liangcai Su, Jiayi Zhang, Chenxiong Qian<sup>†</sup>
</div>
<div align="center">
{roniny, brucejiayi}@connect.hku.hk, {yek21, sulc}@hku.hk, cqian@cs.hku.hk
</div>

<div align="center">
The University of Hong Kong
</div>
<br>
<div align="center">
<sup>*</sup>Equal Contribution
<sup>†</sup>Corresponding author
</div>

## 📝Abstract

The gaming industry has experienced remarkable innovation and rapid growth in recent years. However, this progress has been accompanied by a concerning increase in First-person Shooter game cheating, with aimbots being the most prevalent and harmful tool. Visual aimbots, in particular, utilize game visuals and integrated visual models to extract game information, providing cheaters with automatic shooting abilities. Unfortunately, existing anti-cheating methods have proven ineffective against visual aimbots. To combat visual aimbots, we introduce the first proactive defense framework against visual game cheating, called *Invisibility Cloak*. Our approach adds imperceptible perturbations to game visuals, making them unrecognizable to AI models. We conducted extensive experiments on popular games CrossFire (CF) and Counter-Strike 2 (CS2), and our results demonstrate that *Invisibility Cloak* achieves real-time re-rendering of high-quality game visuals while effectively impeding various mainstream visual cheating models. By deploying *Invisibility Cloak* online in both CF and CS2, we successfully eliminated almost all aiming and shooting behaviors associated with aimbots, significantly enhancing the gaming experience for legitimate players.

## 📺Demo

You can explore the demos of the Invisibility Cloak at https://inviscloak.github.io/.

## 📊Dataset

The [dataset](https://drive.google.com/file/d/1MDqzO62xe4-qrpcOfdCEb5oBHq_Q783v/view?usp=sharing) used in this project is available for download. It includes images from all relevant scenes, along with a script to process the data.

## 📖Citation

If you find this benchmark helpful for your research, please cite our paper:
```bib
@inproceedings{sun2024invisibility,
  title={Invisibility Cloak: Proactive Defense Against Visual Game Cheating},
  author={Sun, Chenxin and Ye, Kai and Su, Liangcai and Zhang, Jiayi and Qian, Chenxiong},
  booktitle={33rd USENIX Security Symposium (USENIX Security 24)},
  pages={3045--3061},
  year={2024}
}

## 📘Introduction

The repository introduces the *Invisibility Cloak*, a project designed to create Cloaks that provide protection to game scene samples by adding adversarial noise. This artifact demonstrates a subset of the experimental data and samples described in the paper. While the same operations can be applied to CF, this artifact only showcases the demos and experiments related to CS2.

## 🔧Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

* Python 3.8 or higher 
* Conda (recommended) or Python virtual environment 
* GPU with CUDA 12.5.0 support (required if using Docker) or other CUDA versions (if not using Docker)


### Using Docker

We provide a pre-configured Docker image file. Ensure you have Docker installed on your system.

1. **Load and run the Docker container:**

   Download the Docker image file [invisicloak_configured.tar](https://drive.google.com/file/d/1FcUy_LG8ySqxaLJAb1_W_PWFeucLT1gA/view?usp=sharing) and load it using the following command:

   ```bash
   docker load -i invisicloak_configured.tar
   ```

   Then, run the Docker container with GPU support:

   ```bash
   docker run -it --rm --gpus all invisicloak:configured
   ```

2. **Activate the conda environment:**

   The Docker container already includes CUDA 12.5.0 and the *invisicloak* conda environment pre-installed. To activate the environment, run:

   ```bash
   conda activate invisicloak
   ```

### Without Docker

If you prefer not to use Docker, follow these steps to set up the environment manually:

1. **Create and activate a conda environment:**
    ```bash
    conda create --name invisicloak python=3.8
    conda activate invisicloak
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### Get Cloaks for the chosen demo scenario in CS2:

To generate cloaked samples for a chosen demo scenario in the CS2 game, use the following command:

```bash
python get_cloak.py --scenario stand
```

The script will generate Cloaks and add them to the input scenario samples, creating protected images.

If you want to visualize the GIF of the cloaked samples, use the following command:

```bash
python get_cloak.py --scenario stand --visualize_gif 1
```

### Command Line Arguments

- `--n_iter`: 
  - Description: Number of iterations.
  - Default: `5`
  - Type: `int`

- `--lr`: 
  - Description: Learning rate for the optimizer.
  - Default: `0.005`
  - Type: `float`

- `--epsilon`: 
  - Description: The L_infinity norm constraint, input value 8 represents 8/255.
  - Default: `8`
  - Type: `int`

- `--local_model`: 
  - Description: Local model to use. Choices are 'yolov5n', 'yolov5s', 'yolov5m'.
  - Default: `yolov5n`
  - Type: `str`

- `--target_model`: 
  - Description: Target cheating model to defend. Choices are 'yolov5n', 'yolov5s', 'yolov5m'.
  - Default: `yolov5n`
  - Type: `str`

- `--gpu`: 
  - Description: GPU device to use for computation.
  - Default: `0`
  - Type: `str`

- `--use_universal_cloak`: 
  - Description: Flag to determine whether to use Universal Cloak (1: yes, 0: no).
  - Default: `1`
  - Type: `int`

- `--visualize_gif`: 
  - Description: Flag to determine whether to generate a GIF (1: yes, 0: no).
  - Default: `0`
  - Type: `int`

- `--scenario`: 
  - Description: CS2 demo scenario to use. Options include:
    - `2people`: Scenario with two people present.
    - `back`: Scenario where the character is facing backwards.
    - `cover`: Scenario where the character is behind cover.
    - `fire`: Scenario where the character is firing a weapon.
    - `flash`: Scenario involving a flash effect.
    - `football`: Scenario with a football next to the character.
    - `halfbody`: Scenario showing half of the character's body.
    - `hited`: Scenario where the character is hit.
    - `jump`: Scenario where the character is jumping.
    - `knife`: Scenario where the character is holding a knife.
    - `op`: Scenario where the character is holding a sniper.
    - `props`: Scenario where the character is hurt by a prop.
    - `reload`: Scenario where the character is reloading a weapon.
    - `run`: Scenario where the character is running.
    - `side`: Scenario showing a side view of the character.
    - `smoke`: Scenario involving smoke.
    - `stand`: Scenario where the character is standing.
    - `usingprop`: Scenario where the character is using a prop.
  - Default: `stand`
  - Type: `str`

## 📁Checking the Results

The results can be found in the `result` directory. The structure of the `result` directory is as follows:

```markdown
.
└── result/
    ├── log/
    │   └──cs2
    │      └──logfile
    └── visualization
        └──cs2_demo
            ├── 2people
            │   ├── attack
            │   ├── gif
            │   └── gt
            ├── back
            │   ├── attack
            │   ├── gif
            │   └── gt
            └── other scenarios...
```

- **log/cs2/logfile**: Contains logs of the experiments, including details of the defense performance against visual aimbots.
- **visualization/cs2_demo/scenarios/attack**: Contains images showing the detection results by the target cheating model on samples protected by the Cloak.
- **visualization/cs2_demo/scenarios/gif**: Contains generated GIFs that simulate actual dynamic game scenes with Cloak protection.
- **visualization/cs2_demo/scenarios/gt**: Contains images showing the detection results by the target cheating model on samples without Cloak protection.
