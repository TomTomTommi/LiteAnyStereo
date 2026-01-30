# Lite Any Stereo: Efficient Zero-Shot Stereo Matching

[`Paper`](https://arxiv.org/abs/2511.16555) [[`Project`](https://tomtomtommi.github.io/LiteAnyStereo/)]


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
