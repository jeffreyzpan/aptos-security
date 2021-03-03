# APTOS Security — A Study of Data-Processing-Based Defenses for Medical Diagnosic Models

## Motivation

Deep learning has proven remarkably effective in diagnosing patients with diabetic retinopathy, a complication of diabetes that is the leading cause of blindness in working-age adults. The [APTOS 2019 Kaggle challenge](https://www.kaggle.com/c/aptos2019-blindness-detection/overview)  highlighted the potential of deep neural networks to accurately diagnose diabetic retinopathy with near-human accuracy.

However, given recent work in adversarial machine learning highlighting the vulnerability of DNNs to adversarial attacks, it's important to ensure that medical models are robust from these sorts of vulnerabilities, given that patient outcomes are at stake. 

This work attempts to measure how robust medical diagnostic models are against current adversarial attacks. We propose using data preprocessing as a cost-effective defense solution.

## Install

This code requires the following dependencies:

- PyTorch >= 1.2.0
- Torchvision >= 0.4.0
- ART >= 1.2.0
- Foolbox >= 3.2.1
- OpenCV >= 3.4.7
- Skopt >= 0.8.1

Additionally, you'll need to download the [APTOS 2019 challenge data](https://www.kaggle.com/c/aptos2019-blindness-detection/overview) from [Kaggle](kaggle.com). Either login on the Kaggle website, join the competition, and download the dataset there directly, or use the useful [Kaggle API](https://github.com/Kaggle/kaggle-api) to join the competition directly from the command line:
```
pip install kaggle
```

Then create an API token via your Kaggle account page (`https://www.kaggle.com/<username>/account`) and move the downloaded token to the location `~/.kaggle/kaggle.json`. Then, ensure that other users can't read this token, since it's your secret token.

```
chmod 600 ~/.kaggle/kaggle.json
```

Finally, you should be able to use the Kaggle API:

```
kaggle competitions download aptos2019-blindness-detection
```

For more Kaggle API instructions and details, go [here](https://github.com/Kaggle/kaggle-api). 

## Attacks and Defenses

Currently, this repo supports the following adversarial attack methods:

- [FGSM](https://arxiv.org/abs/1412.6572)
- [BIM](https://arxiv.org/abs/1607.02533)
- [PGD](https://arxiv.org/abs/1706.06083)
- [DeepFool](https://arxiv.org/abs/1511.04599)

## Usage

To train models from scratch, benchmark adversarial attacks, or run the image contrast optimization, use the scripts in the `scripts` directory.

To train an APTOS model from scratch:

```
GPU=0,1,2,3
INPUT_SIZE=456
CONTRAST=1 (leave as 1 for default contrast)

./scripts/aptos_scratch.sh GPU INPUT_SIZE CONTRAST
```

To resume training:

```
GPU=0,1,2,3 (GPU IDs)
INPUT_SIZE=456 (image size)
CONTRAST=1 (leave as 1 for default contrast)
RESUME=/path/to/saved_model.pth

./scripts/aptos_scratch.sh GPU INPUT_SIZE CONTRAST RESUME
```

To benchmark an adversarial attack method:

```
GPU=0,1,2,3
ATTACK=fgsm from {fgsm,bim,pgd,deepfool}
CONTRAST=1
SIZE=456

./scripts/attack_aptos.sh GPU ATTACK CONTRAST SIZE
```

To run the image contrast optimization:

```
GPU=0,1,2,3
ATTACK=fgsm (note: attack to defend against)
CONTRAST=1 (note: contrast model is originally trained on)
SIZE=456

./scripts/run_contrast.sh GPU ATTACK CONTRAST SIZE
```