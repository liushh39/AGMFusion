<h1 align="center">🔥AGMFusion: A Real-Time End-to-End Infrared and Visible Image Fusion Network Based on Adaptive Guidance Module</h1>

<div align='center'>
    <a href='https://github.com/liushh39' target='_blank'><strong>Shenghao Liu</strong></a><sup> 1</sup>&emsp;
    <a target='_blank'><strong>Xiaoxiong Lan</strong></a><sup> 1</sup>&emsp;
    <a target='_blank'><strong>Wenyong Chen</strong></a><sup> 2</sup>&emsp;
    <a target='_blank'><strong>Zhiyong Zhang</strong></a><sup> 1</sup>&emsp;
    <a target='_blank'><strong>Changzhen Qiu</strong></a><sup> 1†</sup>&emsp;
</div>

<div align='center'>
    <sup>1 </sup>Sun Yat-Sen University&emsp; <sup>2 </sup>Yiren Technology&emsp; <small><sup>†</sup> Corresponding author</small>
</div>


<h1 align="center"><img src="https://github.com/liushh39/AGMFusion/blob/main/img/show.gif" width="800"></h1>

## 🔥 Updates
- **`2024/08/13`**: ✨ We released the paper on [TechRxiv](https://www.techrxiv.org/users/813809/articles/1215064-agmfusion-a-real-time-end-to-end-infrared-and-visible-image-fusion-network-based-on-adaptive-guidance-module).
- **`2024/08/03`**: 🤗 We added a description of the dataset and how to make your own dataset.
- **`2024/07/21`**: 😊 We released the initial version of the code and models. Continuous updates, stay tuned!

## Introduction 📖
This repo, named **AGMFusion**, contains the official PyTorch implementation of our paper [AGMFusion: A Real-Time End-to-End Infrared and Visible Image Fusion Network Based on Adaptive Guidance Module](https://ieeexplore.ieee.org/document/10605610).
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.

If you are interested in image fusion, you can visit our [IVIF-Code-Interpretation](https://github.com/liushh39/IVIF-Code-Interpretation) to read articles with **open-source code**.
## Getting Started 🏁
### 1. Clone the code and prepare the environment
```bash
git clone https://github.com/liushh39/AGMFusion.git
cd AGMFusion

# create env using conda
conda create -n AGMFusion python==3.7.3
conda activate AGMFusion

# install dependencies with pip
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### 2. Train
Download the [dataset](https://pan.baidu.com/s/1PBb-d0mfr1caUKGZGGZMsQ?pwd=udtb), put it in `dataset`
```
├── AGMFusion/dataset
|     ├──train
|         ├── img1
|         ├── img2
|         ├── vi
|         ├── ir
```
It does not matter if there are blank images in the img2 folder of the downloaded dataset.

Or you can create your dataset by running `python create_dataset.py`
```bash
python train.py
```
### 3. Inference 🚀

```bash
python test.py
```

If the script runs successfully, you will get fusion results named in `fusion_results`.

Or, you can change the parameters:

```bash
# more options to see
python test.py -h
```

## Contact Informaiton
If you have any questions, please feel free to contact me at liushh39@mail2.sysu.edu.cn.

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
