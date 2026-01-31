<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
Lite Any Stereo: Efficient Zero-Shot Stereo Matching<h1>      
<p align='center' style="text-align:center;">
    <a href="https://tomtomtommi.github.io/" target="_blank" style="text-decoration: none;">Junpeng Jing</a>,&nbsp;
    <a href="https://scholar.google.com/citations?user=2Y0-0C8AAAAJ&hl=en" target="_blank" style="text-decoration: none;">Weixun Luo</a>,&nbsp;
    <a href="https://yebulabula.github.io/" target="_blank" style="text-decoration: none;">Ye Mao</a>,&nbsp;
    <a href="https://www.imperial.ac.uk/people/k.mikolajczyk"  target="_blank" style="text-decoration: none;">Krystian Mikolajczyk</a>&nbsp;<br/>
&nbsp;Imperial College London<br/>

<div align="center">
  <a href="https://arxiv.org/abs/2511.16555" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
  <a href="https://tomtomtommi.github.io/LiteAnyStereo/" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/Project-Page-9cf" alt="Project Page"></a>
</div>
</p>


![Reading](./assets/fig1.png)

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


## MACs
To compute the model complexity (MACs), use:
```
python flops_count.py
```

## Citation 
If you find this work useful, please consider citing:
```
@article{jing2025lite,
  title={Lite Any Stereo: Efficient Zero-Shot Stereo Matching},
  author={Jing, Junpeng and Luo, Weixun and Mao, Ye and Mikolajczyk, Krystian},
  journal={arXiv preprint arXiv:2511.16555},
  year={2025}
}
```
