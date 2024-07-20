<h1 align="center">🔥AGMFusion: A Real-Time End-to-End Infrared and Visible Image Fusion Network Based on Adaptive Guidance Module</h1>

<div align='center'>
    <a href='https://github.com/liushh39' target='_blank'><strong>Shenghao Liu</strong></a><sup> 1</sup>&emsp;
    <a target='_blank'><strong>Xiaoxiong Lan</strong></a><sup> 1</sup>&emsp;
    <a target='_blank'><strong>Wenyong Chen</strong></a><sup> 2</sup>&emsp;
    <a target='_blank'><strong>Zhiyong Zhang</strong></a><sup> 1</sup>&emsp;
    <a target='_blank'><strong>Changzhen Qiu</strong></a><sup> 1†</sup>&emsp;
</div>

<div align='center'>
    <sup>1 </sup>Sun Yat-Sen University&emsp; <sup>2 </sup>Yiren Technology&emsp; <small><sup>†</sup> Corresponding author</small>;
</div>


<h1 align="center"><img src="https://github.com/liushh39/AGMFusion/blob/main/img/show.gif" width="600"></h1>


## Introduction 📖
This repo, named **AGMFusion**, contains the official PyTorch implementation of our paper [AGMFusion: A Real-Time End-to-End Infrared and Visible Image Fusion Network Based on Adaptive Guidance Module](https://ieeexplore.ieee.org/document/10605610).
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.

## Getting Started 🏁
### 1. Clone the code and prepare the environment
```bash
git clone https://github.com/KwaiVGI/LivePortrait
cd LivePortrait

# create env using conda
conda create -n AGMFusion python==3.7.3
conda activate AGMFusion

# install dependencies with pip
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### 2. Train

### 3. Inference 🚀

```bash
python test.py
```

If the script runs successfully, you will get fusion results named in "fusion_results".

Or, you can change the input by specifying the `-s` and `-d` arguments:

```bash
# more options to see
python test.py -h
```

## Citation 💖
If you find AGMFusion useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@ARTICLE{10605610,
  author={Liu, Shenghao and Lan, Xiaoxiong and Chen, Wenyong and Zhang, Zhiyong and Qiu, Changzhen},
  journal={IEEE Sensors Journal}, 
  title={AGMFusion: A Real-Time End-to-End Infrared and Visible Image Fusion Network Based on Adaptive Guidance Module}, 
  year={2024},
  keywords={Adaptive guidance module;deep learning;image fusion;infrared and visible images},
  doi={10.1109/JSEN.2024.3426274}}
```
