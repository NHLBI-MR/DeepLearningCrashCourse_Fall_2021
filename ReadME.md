# Deep Leanring Crash Course
### Fall 2021

## Overview

Deep learning (DL) is currently the most effective way to model the non-structured data, such as images, videos and time series signals. While there are many excellent online resources to teach deep learning with different levels of complexity, it is still of interests to develop and offer a deep learning crash couse for NHLBI DIR community. The motivation behind this course includes:

  1. Introduce the basics of deep learning for convolution neural network, recurent neural network and generative model.
  2. Introduce practical techniques for model training, debugging, testing, deployment
  3. Help everyone be comfort with coding up their models
  4. Grow interests inside DIR and improve community awareness of applying DL in biomedical R&D
  5. Provide oppertunity for social gathering by bringing colleagues who are interested in these topics together
  6. Prepare trainnees and fellows for DL related job oppertunties

## Instructors

Hui Xue | David C. Hansen
------------ | -------------
![Hui Xue](https://media-exp1.licdn.com/dms/image/C4D03AQGJuEIlujdiHQ/profile-displayphoto-shrink_200_200/0/1597325607723?e=1620259200&v=beta&t=pXeEwCeXeMfOWCfhAps-eJJ8Qrc_Ok6ql7Gp9skanNg) | ![David Hansen](https://media-exp1.licdn.com/dms/image/C5603AQGMGQ5JOiBsGg/profile-displayphoto-shrink_200_200/0/1516604405465?e=1620259200&v=beta&t=_pjDf9VYzkqg1BFVjGzuygr3OKts2m_adFCMTPvpgMw)
hui.xue@nih.gov | davidchansen@gradientsoftware.net
Hui is an active researcher on developing deep learning based cardiac imaging applications. The AI imaging products he built are deployed globally and used daily.  | David is the founder and technical director of Gradient Software. He developed AI solutions for onocoloy planning, image denoising and enhancement, and reporting. 

## Assignments

The assignments and lab contents are the meat of this course. The goal is to demonstrate practical skills to solve real-world DL problems using neural networks and to teach useful software packages and tools for effective model development.

After each lecture, there are suggested reading lists. Please **read** them.

There are 7 assignments in this offering. 

Assignments  | What it is for
------------ | -------------
Setup and basic NN | Set up the development environment, including GPU. Code up a NN with MLP and CNN in numpy, for forward-pass, backprop, optimization and training loop etc.
Pytorch model building | Build up the CNN model using pytorch, implement training and validation
CNN for segmentation and detection | Build up the full-size u-net model for segmentation and detection; Resnet; Add your own loss function
RNN for trigger detection | Build up RNN model for trigger detection from time signal; Solve the problem with LSTM, GRU and Transfomer
GAN and fun | Build up the generative adversial network to create new cardaic images; Will your image be good enough to let your model segment it (from assigment 3)?
Data management, experiment manage | Use DVC to manage data and training models; Add wandb for experiment management
Mode deployemnt | Use streamlit to deploy your model from assgiment 2; You can run the model as a web service and show to your lab

After going through the assignments, you should feel comfort/confident to build, debug and deploy your own models! :clap::clap::clap:

## *Schedule*

Time     | Lectures                          | What it is for   |  Reading list     | Assignments
---------| --------------------------------- | ---------------- | --------------    | ------------
L1       | Fundation of deep learning        | Building blocks, MLP, course overview                          |                   | 
L2       | Fundation of deep learning (cont.)| backprop, optimization, loss function ...                      |                   | Assignment 1
L3       | Convolution neural network        | Convolution and its variants, badkprop of conv, Common architectures|              | Assignment 2
L4       | Convolution neural network (cont.)| Resnet, u-net, use CNN for classification, segmentation, detection|                | Assignment 3
L5       | Recurrent neural network          | RNN basics, vallina RNN, LSTM                                  |                   | 
L6       | RNN and Transformer               | GRU, Self-attention, Workd embeding, Transformers              |                   | Assignment 4
L7       | Model training                    | Nut and bolts, parallel training, learning rate scheduling, OneCycle, Debug       |                   | 
L8       | Generative  model                 | Generative vs. discriminative, GAN, its cost function, GAN variants, CycleGAN, CycleGAN+Unet        |                   |  Assignment 5
L9       | Tooling and infrastructures       | Data management, experiment management, testing, tooling        |                   |  Assignment 6
L10      | Other topics                      | Latest progress, meta-learning, model deployment, build your own DL lab        |                   |  Assignment 7
L11      | Invited lecture 1                 | Weak supervision and software 2.0        |                   |  
L12      | Invited lecture 2                 | Full stack deep learning        |                   |  
