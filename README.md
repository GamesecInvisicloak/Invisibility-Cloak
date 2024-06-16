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

<!-- ## 📢News -->

## 📝Abstract

The gaming industry has experienced remarkable innovation and rapid growth in recent years. However, this progress has been accompanied by a concerning increase in First-person Shooter game cheating, with aimbots being the most prevalent and harmful tool. Visual aimbots, in particular, utilize game visuals and integrated visual models to extract game information, providing cheaters with automatic shooting abilities. Unfortunately, existing anti-cheating methods have proven ineffective against visual aimbots. To combat visual aimbots, we introduce the first proactive defense framework against visual game cheating, called *Invisibility Cloak*. Our approach adds imperceptible perturbations to game visuals, making them unrecognizable to AI models. We conducted extensive experiments on popular games CrossFire (CF) and Counter-Strike 2 (CS2), and our results demonstrate that *Invisibility Cloak* achieves real-time re-rendering of high-quality game visuals while effectively impeding various mainstream visual cheating models. By deploying *Invisibility Cloak* online in both CF and CS2, we successfully eliminated almost all aiming and shooting behaviors associated with aimbots, significantly enhancing the gaming experience for legitimate players.

## 📺Demo

You can explore the demos of the Invisibility Cloak at https://inviscloak.github.io/.

## 📘 Introduction

The repository introduces the *Invisibility Cloak*, a project designed to create Cloaks that provide protection to game scene samples by adding adversarial noise. 

## 🔧Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

* Python 3.8 or higher
* Conda (recommended) or Python virtual environment
* GPU with CUDA support (optional but recommended for faster processing)

### Installation

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

#### Get Cloaks for choosen demo scenario in CS2 game
To generate cloaked samples for a chosen demo scenario in the CS2 game, use the following command:

```
python get_cloak.py --scenario stand
```

The script will generate Cloaks and add them to the input scenario samples, creating protected images.

If you want to visualize the GIF of the cloaked samples, use the following command:

```
python get_cloak.py --scenario stand --visualize_gif 1
```

### Command Line Arguments

- `--game`: Dataset to use. Default is 'cs2'.
- `--n_iter`: Number of iterations for the attack. Default is 5.
- `--lr`: Learning rate for the optimizer. Default is 0.005.
- `--epsilon`: Attack strength. Default is 8.
- `--local_model`: Local model to use ('yolov5n', 'yolov5s', 'yolov5m'). Default is 'yolov5n'.
- `--target_model`: Target cheating model to defend ('yolov5n', 'yolov5s', 'yolov5m'). Default is 'yolov5n'.
- `--gpu`: GPU device to use. Default is '0'.
- `--use_global`: Whether to use global noise (1: yes, 0: no). Default is 1.
- `--visualize_gif`: Whether to visualize the GIF (1: yes, 0: no). Default is 0.
- `--scenario`: CS2 demo scenario to use. Default is 'cover'. Options include '2people', 'back', 'cover', 'fire', 'flash', 'football', 'halfbody', 'hited', 'jump', 'knife', 'op', 'props', 'reload', 'run', 'side', 'smoke', 'stand', 'usingprop'.

