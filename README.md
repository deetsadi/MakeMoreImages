# MakeMoreImages
## Description
Inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore), a simple autoregressive character model, **MakeMoreImages** is a one script implementation of a Denoising Diffusion Probabilistic Model (DDPM) from scratch written clearly and concisely for educational purposes. Train and generate images that look like your data with just one command.  

With a focus on the demonstrating the basic concepts of Diffusion models, this script takes a first principles approach to building the model without some of the tweaks introduced in later papers, such as [cosine noise scheduling](https://arxiv.org/abs/2105.05233). As such, it can easily be used as a baseline for projects or research that rely on a DDPM backbone. 
To learn more about the theory behind Diffusion models, check out [this](https://adityas03.medium.com/diffusion-models-from-ground-up-2d41de6ca0a6) blog post (shameless self promotion).  

  
The DDPM paper can be found [here](https://arxiv.org/pdf/2006.11239.pdf). There are several comments in the code which refer the reader to equations and algorithms shown in the paper to enhance readability and understanding.

## Usage
Install requirements using `pip install -r requirements.txt`. 
Use `python ddim.py -h` to see a full list of hyperparameters you can tweak if necessary, such as batch size. Its recommended to keep your image size small (256 at most) to speed up training and inference.
### Train
`python ddim.py --in_dir <path to image folder> --image_size <desired size of output images>`.  
### Inference
`python ddim.py --sample --model <path to saved model>`. 

