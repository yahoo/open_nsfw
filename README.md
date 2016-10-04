# Open nsfw model
This repo contains code for running Not Suitable for Work (NSFW) classification deep neural network Caffe models. Please refer our [blog](https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for) post which describes this work and experiments in more detail.

#### Not suitable for work classifier
Detecting offensive / adult images is an important problem which researchers have tackled for decades. With the evolution of computer vision and deep learning the algorithms have matured and we are now able to classify an image as not suitable for work with greater precision.

Defining NSFW material is subjective and the task of identifying these images is non-trivial. Moreover, what may be objectionable in one context can be suitable in another. For this reason, the model we describe below focuses only on one type of NSFW content: pornographic images. The identification of NSFW sketches, cartoons, text, images of graphic violence, or other types of unsuitable content is not addressed with this model.

Since images and user generated content dominate the internet today, filtering nudity and other not suitable for work images becomes an important problem. In this repository we opensource a Caffe deep neural network for preliminary filtering of NSFW images. 

![Demo Image](https://66.media.tumblr.com/a24135a56ecf20d7efb81dda0f4ccbac/tumblr_inline_oebl0iNWRM1rilvr1_500.png "")


#### Usage

* The network takes in an image and gives output a probability (score between 0-1) which can be used to filter not suitable for work images. Scores < 0.2 indicate that the image is likely to be safe with high probability. Scores > 0.8 indicate that the image is highly probable to be NSFW. Scores in middle range may be binned for different NSFW levels. 
* Depending on the dataset, usecase and types of images, we advise developers to choose suitable thresholds. Due to difficult nature of problem, there will be errors, which depend on use-cases / definition / tolerance of NSFW.  Ideally developers should create an evaluation set according to the definition of what is safe for their application, then fit a [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve to choose a suitable threshold if they are using the model as it is. 
* ***Results can be improved by [fine-tuning](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html)*** the model for your dataset/ uscase / definition of NSFW. We do not provide any guarantees of accuracy of results. Please read the disclaimer below.
* Using human moderation for edge cases in combination with the machine learned solution will help improve performance.

#### Description of model
We trained the model on the dataset with NSFW images as positive and SFW(suitable for work) images as negative. These images were editorially labelled. We cannot release the dataset or other details due to the nature of the data. 

We use [CaffeOnSpark](https://github.com/yahoo/CaffeOnSpark) which is a wonderful framework for distributed learning that brings deep learning to Hadoop and Spark clusters for training models for our experiments. Big thanks to the CaffeOnSpark team!

The deep model was first pretrained on ImageNet 1000 class dataset. Then we finetuned the weights on the NSFW dataset.
We used the thin resnet 50 1by2 architecture as the pretrained network. The model was generated using [pynetbuilder](https://github.com/jay-mahadeokar/pynetbuilder) tool and replicates the [residual network](https://arxiv.org/pdf/1512.03385v1.pdf) paper's 50 layer network (with half number of filters in each layer).  You can find more details on how the model was generated and trained [here](https://github.com/jay-mahadeokar/pynetbuilder/tree/master/models/imagenet)

Please note that deeper networks, or networks with more filters can improve accuracy. We train the model using a thin residual network architecture, since it provides good tradeoff in terms of accuracy, and the model is light-weight in terms of runtime (or flops) and memory (or number of parameters).

#### Docker Quickstart
This Docker quickstart guide can be used for evaluating the model quickly with minimal dependency installation.

Install Docker Engine
- [Windows Installation](https://docs.docker.com/v1.8/installation/windows/)
- [Mac OSX Installation](https://docs.docker.com/v1.8/installation/mac/)
- [Ubuntu Installation](https://docs.docker.com/v1.8/installation/ubuntulinux/)

Build a caffe docker image (CPU) 
```
docker build -t caffe:cpu https://raw.githubusercontent.com/BVLC/caffe/master/docker/standalone/cpu/Dockerfile
```

Check the caffe installation
```
docker run caffe:cpu caffe --version
caffe version 1.0.0-rc3
```

Run the docker image with a volume mapped to your `open_nsfw` repository. Your `test_image.jpg` should be located in this same directory.
```
cd open_nsfw
docker run --volume=$(pwd):/workspace caffe:cpu \
python ./classify_nsfw.py \
--model_def nsfw_model/deploy.prototxt \
--pretrained_model nsfw_model/resnet_50_1by2_nsfw.caffemodel \
test_image.jpg
```

We will get the NSFW score returned:
```
NSFW score:   0.14057905972
``` 
#### Running the model
To run this model, please install [Caffe](https://github.com/BVLC/caffe) and its python extension and make sure pycaffe is available in your PYTHONPATH.

We can use the [classify.py](https://github.com/BVLC/caffe/blob/master/python/classify.py) script to run the NSFW model. For convenience, we have provided the script in this repo as well, and it prints the NSFW score. 

 ```
 python ./classify_nsfw.py \
 --model_def nsfw_model/deploy.prototxt \
 --pretrained_model nsfw_model/resnet_50_1by2_nsfw.caffemodel \
 INPUT_IMAGE_PATH 
 ```
 
#### ***Disclaimer***
The definition of NSFW is subjective and contextual. This model is a general purpose reference model, which can be used for the preliminary filtering of pornographic images. We do not provide guarantees of accuracy of output, rather we make this available for developers to explore and enhance as an open source project. Results can be improved by [fine-tuning](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html) the model for your dataset.

#### License
Code licensed under the [BSD 2 clause license] (https://github.com/BVLC/caffe/blob/master/LICENSE). See LICENSE file for terms.

#### Contact
The model was trained by [Jay Mahadeokar](https://github.com/jay-mahadeokar/),  in collaboration with [Sachin Farfade](https://github.com/sachinfarfade/) , [Amar Ramesh Kamat](https://github.com/amar-kamat), [Armin Kappeler](https://github.com/akappeler) and others. Special thanks to Gerry Pesavento for taking the initiative for open-sourcing this model. If you have any queries, please raise an issue and we will get back ASAP.

