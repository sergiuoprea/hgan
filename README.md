<div align="center">    
 
# HandGAN - The power of GANs in your Hands!

[![Paper](http://img.shields.io/badge/preprint-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2103.15017)
[![Conference](https://img.shields.io/badge/IJCNN-2021-blue.svg)](https://arxiv.org/abs/2103.15017)

![Architecture](assets/architecture.png)
 
</div>
 
## Description   

HandGAN (H-GAN) is a cycle-consistent adversarial approach designed to translate synthetic images of hands to the real domain. Synthetic hands provide complete ground-truth annotations, yet they do not approximate the underlying distribution of real images of hands. The goal is to brdige the gap between the synthetic and real domain at a distribution level. We strive to provide the perfect blend of a realistic hand appearance with precise synthetic annotations. H-GAN is able to translate synthetic hands so as they appearance shares similarities with images of hands from the real domain. It maintains the hand shape during the translation to ensure the correspondance with the ground-truth annotations.

Feel free to use our HandGAN to level up your synthetic images of hands. Your deep learning-based models are data hungry, so you just need to feed them properly!

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/sergiuoprea/hgan.git

# install project   
cd hgan
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to the code folder.   
 ```bash
# code folder
cd code

# run main.py 
python main.py    
```

## Publication
<div align="center">

 ### [**H-GAN: The power of GANs in your Hands**](https://arxiv.org/abs/2103.15017)
 
 International Joint Conference on Neural Networks (IJCNN) - 2021
 
 *Sergiu Oprea, Giorgos Karvounas, Pablo Martínez-González, Nikolaos Kyriazis, Sergio Orts-Escolano, Iason Oikonomidis, Alberto García-García, Aggeliki Tsoli, José García-Rodríguez, and Antonis Argyros*

</div>

### How to cite this work?
If you use HandGAN, please cite:
```
@article{Oprea2021,
  author    = {Sergiu Oprea and Giorgos Karvounas and Pablo Martinez{-}Gonzalez and Nikolaos Kyriazis and Sergio Orts{-}Escolano and Iason Oikonomidis and Alberto Garcia{-}Garcia and Aggeliki Tsoli and Jos{\'{e}} Garc{\'{\i}}a Rodr{\'{\i}}guez} and Antonis A. Argyros},
  title     = {{H-GAN:} the power of GANs in your Hands},
  journal   = {CoRR},
  volume    = {abs/2103.15017},
  year      = {2021}
}
```

For any inquiries, feel free to create an issue or contact Sergiu Oprea ([soprea@dtic.ua.es](mailto:soprea@dtic.ua.es)).
