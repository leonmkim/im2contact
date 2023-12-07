# Im2Contact: Vision-Based Contact Localization Without Touch or Force Sensing
[[Paper]](https://openreview.net/pdf?id=h8halpbqB-) [[Project page and videos]](https://sites.google.com/view/im2contact/home) 

## Instructions
Note: Code and dependencies have not been tested on different setups. For any issues, please reach out to me at `leonmkim@seas.upenn.edu` or create a github issue.

Simulation data generation code is not yet included. Feel free to reach out and give me a kick in the butt to get me going on this. 

### Dependencies
Dependencies can be installed using `pip install -r requirements.txt`

Some submodules are needed which can be included using recursive git cloning. 

### Training
Files under `./config` can be used to modify training parameters. 

To run the training script:
```
python training.py fit --data ./config/training.yaml --data ./config/curated_valid_real_datasets.yaml --trainer ./config/online_training.yaml
```

## BibLaTeX
```
@inproceedings{kimIm2ContactVisionBasedContact2023,
	title = {Im2Contact: Vision-Based Contact Localization Without Touch or Force Sensing},
	url = {https://openreview.net/forum?id=h8halpbqB-},
	shorttitle = {Im2Contact},
	abstract = {Contacts play a critical role in most manipulation tasks. Robots today mainly use proximal touch/force sensors to sense contacts, but the information they provide must be calibrated and is inherently local, with practical applications relying either on extensive surface coverage or restrictive assumptions to resolve ambiguities. We propose a vision-based extrinsic contact localization task: with only a single {RGB}-D camera view of a robot workspace, identify when and where an object held by the robot contacts the rest of the environment. We show that careful task-attuned design is critical for a neural network trained in simulation to discover solutions that transfer well to a real robot. Our final approach im2contact demonstrates the promise of versatile general-purpose contact perception from vision alone, performing well for localizing various contact types (point, line, or planar; sticking, sliding, or rolling; single or multiple), and even under occlusions in its camera view. Video results can be found at: https://sites.google.com/view/im2contact/home},
	eventtitle = {7th Annual Conference on Robot Learning},
	author = {Kim, Leon and Li, Yunshuang and Posa, Michael and Jayaraman, Dinesh},
	date = {2023-08-30},
	langid = {english},
}
```