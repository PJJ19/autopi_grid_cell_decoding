# AutoPI grid cell decoding
Explanation of the decoding method used for the autopi project

## Table of contents

<li>Installation</li>
<li>
  Notebooks for method explanation:
  <ol>1.Simulating grid cell spike trains</ol>
  <ol>2.Estimating grid orientation and spacing</ol>
  <ol>3.Transforming 2D Cartesian coordinates into 2D toroidal coordinates</ol>
  <ol>4.Predicting toroidal location based on grid cell activity</ol>
  <ol>5.Reconstructing movement path from sequences of predicted toroidal locations</ol>
</li>

## Installation
<li>1. Python environment</li>


Here's a step by step guide to implement the method:

Make sure you have Anaconda or Miniconda installed

First create your conda environment:

```
conda create -n torch python=3.8

conda activate torch
```

<li>2. SpikeA package</li>

Install the SpikeA package

```
cd ~/repo
git clone https://github.com/kevin-allen/spikeA.git
cd ~/repo/spikeA
pip install -e ~/repo/spikeA
cd ~/repo/spikeA/spikeA/
python setup.py build_ext --inplace
```

<li>3. Pytorch </li>

Install [Pytorch](https://pytorch.org/get-started/locally/)

If you have GPU on your computer:

```
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

```

If not:

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch

```
