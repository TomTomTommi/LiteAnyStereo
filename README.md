<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
Lite Any Stereo (LAS) Series<h1>  

<div align="center">
  <a href="https://arxiv.org/abs/2511.16555" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/LAS-Paper%20%28arXiv%29-red" alt="LAS paper arXiv"></a>
  <a href='https://tomtomtommi.github.io/LiteAnyStereo/'><img src='https://img.shields.io/badge/LAS-Project%20Page-deepgreen' alt='LAS Project Page'></a>
  <a href='https://tomtomtommi.github.io/LiteAnyStereoV2/'><img src='https://img.shields.io/badge/LAS%20V2-Project%20Page-blue' alt='LAS V2 Project Page'></a>
</div>
</p>

Welcome to the official repository for the **Lite Any Stereo (LAS) series**. This repository currently contains the released code, checkpoints, demos, and evaluation scripts for **Lite Any Stereo (LAS/V1)**, and will also host the upcoming **Lite Any Stereo V2 (LAS2)** code release.

## Latest Update

**Lite Any Stereo V2: Faster and Stronger Efficient Zero-Shot Stereo Matching** is now finished. Please visit the [LAS V2 project page](https://tomtomtommi.github.io/LiteAnyStereoV2/) for the latest results and interactive comparisons.

The LAS V2 paper and code will be released soon. The V2 code will be added to this repository, so this repo should be used as the code link for the LAS series.

## Lite Any Stereo

Lite Any Stereo (LAS/V1) is a super efficient stereo matching model with strong zero-shot generalization ability. It outperforms or matches accuracy-oriented models that do not use foundational priors, while requiring less than 1% of their computational cost.

## Demo
Several example stereo image pairs are provided in the `/assets/` directory. 

You can visualize zero-shot stereo matching results of Lite Any Stereo on real-world scenes by running:
```
python demo.py
```
You can also test the model on your own stereo image pairs by replacing the input images.

## Checkpoint
Before running the demo, please download the pretrained checkpoints from
 [google drive](https://drive.google.com/drive/folders/1UvDx296pVk7pC2rozKIpQF_EXcOleZOB?usp=sharing) . 
Then place them in: `./checkpoints/`


## Benchmark Results 
To reproduce the benchmark results reported in Table 3 and Table 4 of the paper, run:
```
sh evaluate.sh
```
The results of [Lite-CREStereo++](https://github.com/TomTomTommi/LiteCREStereo_plusplus) can be reproduced here.

## MACs
To compute the model complexity (MACs), use:
```
python flops_count.py
```

##  Runtime
To measure the inference time, run:
```
python profile_speed.py
```
This script uses CUDA synchronization for more accurate latency measurement.
The initial version followed the evaluation practice of previous methods and reported runtime using `evaluate_stereo.py`

## Citation
If you find the released LAS/V1 code useful, please consider citing:
```
@article{jing2025lite,
  title={Lite Any Stereo: Efficient Zero-Shot Stereo Matching},
  author={Jing, Junpeng and Luo, Weixun and Mao, Ye and Mikolajczyk, Krystian},
  journal={arXiv preprint arXiv:2511.16555},
  year={2025}
}
```

The LAS V2 citation will be added when the paper is available.
