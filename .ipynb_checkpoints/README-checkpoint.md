6-part Tutorial on Self Supervised Denoising
=========

This repository contains the codes for the Self-supervised denoising tutorial series.

An initial shortened version of the tutorial was given at Transform22 by:
 - Claire Birnie (claire.birnie@kaust.edu.sa), and 
 - Sixiu Liu (sixiu.liu@kaust.edu.sa).
This was presented as a live-stream event on YouTube on April 27 2022 at 11 UTC and can be viewed [here](https://www.youtube.com/watch?v=d9yv90-JCZ0)

Tutorial overview
---------------------------

Self-supervised learning offers a solution to the common limitation of the lack of noisy-clean pairs of data for training deep learning seismic 
denoising procedures.

In this tutorial series, we will explain the theory behind blind-spot networks and how these can be used in a self-supervised manner, removing any 
requirement of clean-noisy training data pairs. We will deep dive into how the original methodologies for random noise can be adapted to handle  structured noise, in particular trace-wise noise. We conclude by introducing how explainable AI can be used to select the blind-masks required for coherent noise suppression.

If you found the tutorial useful please consider citing our work in your studies:

> Birnie, C., M. Ravasi, S. Liu, and T. Alkhalifah, 2021, The potential of self-supervised networks for random noise 
> suppression in seismic data: Artificial Intelligence in Geosciences.

> Liu, S., C. Birnie, and T. Alkhalifah, 2022, Coherent noise suppression via a self-supervised deep learning scheme: 
> 83rd EAGE Conference and Exhibition 2022, European Association of Geoscientists & Engineers, 1–5

>Birnie, C. and Ravasi, M., 2024. Explainable artificial intelligence‐driven mask design for self‐supervised seismic
>denoising. Geophysical Prospecting.

Repository overview
---------------------------

This repository contains 6 tutorial notebooks alongside a handful of files containing useful python functions. 

Disclaimer: the code has all been written and tested on Linux operating systems, where GPU access is available. Neither of the authors are professional software developers therefore, whilst we have spent significant time testing the code, we cannot guarantee it is free of bugs.

| Tutorial   | Tutorial (Github) | Tutorial (Colab) |
|-----------|------------------|------------------|
| 1: Blind-spot random noise suppression | [Link](Tutorial1-BlindSpotNetwork.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebirnie92/SelfSupervisedDenoising/blob/main/Solutions/Solution_Tutorial1-BlindSpotNetwork.ipynb#scrollTo=33f35f5b)  |
| 2: Field Application - Blind-spot random noise suppression | [Link](Tutorial2-BlindSpotNetwork-FieldApplication.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebirnie92/KAUST-Iraya_SummerSchool2021/blob/main/notebooks/day2_active_learning.ipynb)  |
| 3: Blind-trace coherent noise suppression | [Link](Tutorial3-BlindTraceNetwork.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebirnie92/KAUST-Iraya_SummerSchool2021/blob/main/notebooks/day3_Part_1_Generating_BERT_Embedding.ipynb)  |
| 4: Field Application - Blind-trace coherent noise suppression | [Link](Tutorial4-BlindTraceNetwork-FieldApplication.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebirnie92/KAUST-Iraya_SummerSchool2021/blob/main/notebooks/day3_Part_2_title_generator.ipynb)  |
| 5: Automated Blind-mask design | [Link](Tutorial5-XAIMaskGeneration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebirnie92/KAUST-Iraya_SummerSchool2021/blob/main/notebooks/day3_Part_2_title_generator.ipynb)  |
| 5(half): Automated Blind-mask design | [Link](Tutorial5-XAIMaskGeneration-Quick.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebirnie92/KAUST-Iraya_SummerSchool2021/blob/main/notebooks/day3_Part_2_title_generator.ipynb)  |
| 6: Field Application - Automated Blind-mask design | [Link](Tutorial6-XAIMaskGeneration-FieldApplication.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebirnie92/KAUST-Iraya_SummerSchool2021/blob/main/notebooks/day3_Part_2_title_generator.ipynb)  |


Data 
-----

All data can be obtained from [this One Drive](https://drive.google.com/drive/folders/1AovokXsyo6VYxfzLZ2gpvVHLadh5wuN1?usp=sharing). This tutorial series always introduces a concept and illustrates it on synthetic data before applying it to field data. The field data examples used in this work include a post-stack depth migrated image obtained from the Madagascar repository and a short recording from a seismic-while-drilling acquisition led by Prof. Matteo Ravasi at King Abdullah University of Science and Technology. We are grateful to the owners of these data for making them available.


