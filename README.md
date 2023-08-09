# MakeMoreImages
## Description
Inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore), a simple autoregressive character model, **MakeMoreImages** is a one script implementation of a Denoising Diffusion Probabilistic Model (DDPM) from scratch written concisely for educational purposes. Train and generate images that look like your data with just one command.
With a focus on the demonstrating the basic concepts of Diffusion models, this script isn't for the purpose of training models that achieve spectacular performance. However, it can easily be used as a baseline for projects or research that rely on a DDPM backbone. 
To learn more about the theory behind Diffusion models, check out [this](https://adityas03.medium.com/diffusion-models-from-ground-up-2d41de6ca0a6) blog post (shameless self promotion). The original paper can be found [here](https://arxiv.org/pdf/2006.11239.pdf).

## Usage
Install requirements using `pip install -r requirements.txt`. 
Use `python ddim.py -h` to see a full list of hyper parameters you can tweak if necessary, such as batch size.
### Train
`python ddim.py --in_dir <path to image folder> --image_size <desired size of output images>`.
### Inference
`python ddim.py --sample --model <path to saved model>`. 
