# CFPNet: Improving Lightweight ToF Depth Completion via Cross-zone Feature Propagation
###  [Paper]

The paper has been accepted to 3DV 2025. I will provide the link to the paper once it is available either from the conference or Arxiv.


### Installation
```bash
pip install -r requirements.txt
```

### Prepare the data and pretrained model
Please refer to [DELTAR](https://github.com/zju3dv/deltar) for the data preparation.

Please download the pretrained model from [Baidu Yun](https://pan.baidu.com/s/1wUD3dv-E82oIz5UNjGcpwA) (password: fhpv) and put it in the correct directory and rename it to `best.pth`. 

Specifically, 
```bash
change baseline.pt to best.pt and put it under train_deltar_change_embedding_no_clip_grad_hist_encoder_optimized_10x.pt,
```
```bash
change ours.pt to best.pt and put it under train_deltar_change_embedding_no_clip_grad_hist_encoder_optimized_10x_combine1.pt
```
The resulting structure should look sth like this:
```
deltar
├── data
│   ├── demo
│   └── ZJUL5
│   └── nyu_depth_v2
└── weights
    └── train_deltar_change_embedding_no_clip_grad_hist_encoder_optimized_10x_combine1
        └── best.pt (rename the pretrained model to best.pt)
```

### Command to train on the NYU dataset
```bash
python train.py @configs/train_deltar_change_embedding_no_clip_grad_hist_encoder_optimized_10x_combine1.txt
```


### Command to evaluate on the NYU and ZJUL5 dataset
```bash
python evaluate_all.py @configs/train_deltar_change_embedding_no_clip_grad_hist_encoder_optimized_10x_combine1.txt --selected_epoch best
python evaluate_all.py @configs/train_deltar_change_embedding_no_clip_grad_hist_encoder_optimized_10x_combine1.txt --test_dataset nyu --selected_epoch best
```

Note that we train on the NYU dataset and evaluate on both the NYU and ZJUL5 datasets. The model that perform the best
on NYU dataset will be chosen to evaluate on the ZJUL5 dataset. This follows the same protocol as the DELTAR.

If you do not set the selected_epoch, the code will go through all available epochs and generate an excel file that contains the result for all epochs.



## Citation

If you find this code useful for your research, please use the following BibTeX entry. 


## Acknowledgements

We would like to thank the authors ofnd [DELTAR](https://github.com/zju3dv/deltar) for open-sourcing their projects.

