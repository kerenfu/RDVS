# DCTNet_plus:Salient Object Detection in RGB-D Videos

Code for paper, '**Salient Object Detection in RGB-D Videos**' 

[**Supplementary material** is also attached in the repo]


<p align="center">
    <img src="figures/Overview.png" width="100%" /> <br />
    <em> 
    Overview of DCTNet. (a) shows the big picture. (b) and (c) show the details of MAM and RFM, respectively. In the attention operations on the right-hand side in (c), since the coordinate attention and spatial attention processes are similar, the operations of spatial attention are represented in parentheses and are not repeated.
    </em>
</p>


## Downloads
1. **RDVS dataset.**
   - Full dataset with **realistic depth** (4.84G, 57 sequences): [Baidu Pan]()(fetch code: ) | [Google Drive]() (**Update link:**)
   - Full dataset with synthetic deptn (4.76G, 57 sequences): [Baidu Pan]()(fetch code: ) | [Google Drive]() (**Update link:**)
   - Training Set containing realistic and synthetic depth (2.56G, 32 sequences): [Baidu Pan]()(fetch code: ) | [Google Drive]() (**Update link:**)
   - Test Set containing realistic and synthetic depth (2.30G, 25 sequences): [Baidu Pan]()(fetch code: ) | [Google Drive]() (**Update link:**)
   - **Noting: realistic depth is in "/Depth" and synthetic depth is in "/SyntheticDepth"**
  

2. **DCTNet+ model.**
   - original model ckpt: [Baidu Pan](https://pan.baidu.com/s/1T_f_IPM9cJt4pzWbFTSgQQ?pwd=wm08)(fetch code: wm08) | [Google Drive]()
   - finetune on the test set of RDVS with **realistic depth**: [Baidu Pan](https://pan.baidu.com/s/1zSD_0ZyeSCedcyJcLX0G-w?pwd=wm08)(fetch code: wm08) | [Google Drive]()
   - finetune on the test set of RDVS with **synthetic depth**: [Baidu Pan](https://pan.baidu.com/s/1TpKoZULR-yuAeCvrNVoalg?pwd=wm08)(fetch code: wm08) | [Google Drive]()


3. **our training set and test set**
   - training set: [Baidu Pan]()(fetch code: ) | [Google Drive]()
   - test set: [Baidu Pan]()(fetch code: ) | [Google Drive]()



## Usage
1. Requirements
   - Python
   - PyTorch
   - Torchvision
   - Cuda

2. Training
   - Download the pre_trained ResNet34 backbone: [Baidu Pan]()(fetch code: ) | [Google Drive]() to './model/resnet/pre_train/'.
   - Download the train dataset (containing DAVIS16, DAVSOD, FBMS and DUTS-TR) from [Baidu Pan]()(fetch code: ) | [Google Drive]() and save it at './dataset/train/*'.
   - Download the pre_trained RGB, depth and flow stream models from [Baidu Pan]()(fetch code: ) | [Google Drive]() to './checkpoints/'
     - Noting: the pre_trained RGB should be saved at './checkpoints/spatial', pre_trained depth shoule be saved at './checkpoints/depth' and flow is same. 
   - The training of entire DCTNet is implemented on one NVIDIA RTX 3090 GPU.
     - run  `python train.py` in terminal
3. Testing
   - Download the test data (containing DAVIS16, DAVSOD, FBMS, SegTrack-V2, VOS) from  [Baidu Pan]()(fetch code: ) | [Google Drive]() and save it at './dataset/test/*'
   - Download the trained model from [Downloads](#downloads)(original model ckpt) and modify the  `model_path` to its saving path in the `test.py`.
   - Run `python test.py` in the terminal.

