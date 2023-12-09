---
layout: single
# title: "Research"
permalink: /research/
comments: true
author_profile: true
# classes: wide
toc: true
toc_label: "By Categories"
toc_icon: "book"
toc_sticky: true
---

This page is still under construction. More unpublished/ongoing content will be added soon!


## Research

<table style="width:100%;border:0px;border-spacing:0px;border-collapse:separate;margin-right:auto;margin-left:auto;">
  <tr>
    <td style="padding:2.5%;width:25%;vertical-align:middle;min-width:120px">
      <img src="../assets/images/projects/bliva.png" alt="project image" style="width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="padding:2.5%;width:75%;vertical-align:middle">
      <h3>BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions</h3>
      <a href="https://gordonhu608.github.io/">Wenbo Hu*</a>, <a href="https://yfxu.com/">Yifan Xu*</a>, <a href="https://jerryli1019.github.io">Yi Li</a>, <strong><a href="https://weiyueli7.github.io/">Weiyue Li</a></strong>, <a href="https://zeyuan-chen.com/">Zeyuan Chen</a>, <a href="https://pages.ucsd.edu/~ztu/">Zhuowen Tu</a>
      <br>
      <i>AAAI 2024</i>
      <br>
      <a href="https://gordonhu608.github.io/bliva/">website</a> / <a href="https://arxiv.org/abs/2308.09936">arxiv</a> / <a href="https://github.com/mlpc-ucsd/BLIVA">code</a>
      <p>We introduce BLIVA, an augmented version of InstructBLIP with Visual Assistant. BLIVA incorporates the query embeddings from InstructBLIP and also directly projects encoded patch embeddings into the LLM, a technique inspired by LLaVA. This approach ensures that the model captures intricate details potentially missed during the query decoding process. Empirical evidence demonstrates that our model, BLIVA, significantly enhances performance in processing text-rich VQA benchmarks (up to 17.76% in OCR-VQA benchmark) and in undertaking typical VQA benchmarks (up to 7.9% in Visual Spatial Reasoning benchmark), comparing to our baseline InstructBLIP. BLIVA demonstrates significant capability in decoding real-world images, irrespective of text presence.</p>
    </td>
  </tr>
</table>

## Selective Projects

### AI/ML

<table style="width:100%;border:0px;border-spacing:0px;border-collapse:separate;margin-right:auto;margin-left:auto;">
  <tr>
    <td style="padding:2.5%;width:25%;vertical-align:middle;min-width:120px">
      <img src="../assets/images/projects/cdcgans.png" alt="project image" style="width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="padding:2.5%;width:75%;vertical-align:middle">
      <h3>Generative Vision: Image Synthesis with Conditional DCGANs</h3>
      <strong>Weiyue Li</strong>, Charles Ye
      <br>
      <a href="https://github.com/weiyueli7/cGANs-cDCGANs/blob/main/doc/report.pdf">report</a> / <a href="https://github.com/weiyueli7/cGANs-cDCGANs">code</a>
      <p>Our project proposes two architectures with early or late concatenations to improve the performance of the original cGANs paper's architecture. The project involves training cDCGANs and cGANs on large-scale labeled datasets, where the models are conditioned on auxiliary information such as class labels. We also employ various metrics to evaluate the performance, and the results show that our cDCGANs model validates its effectiveness in generating high-quality and realistic images.</p>
    </td>
  </tr>
  <!-- <tr>
    <td style="padding:2.5%;width:25%;vertical-align:middle;min-width:120px">
      <img src="../assets/images/projects/dcgans.png" alt="project image" style="width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="padding:2.5%;width:75%;vertical-align:middle">
      <h3>DCGANs on Image Classification Task</h3>
      <strong>Weiyue Li</strong>, Yi Li
      <br>
      <a href="https://github.com/weiyueli7/DCGANs/blob/main/report.pdf">report</a> / <a href="https://github.com/weiyueli7/DCGANs">code</a>
      <p>Our project aims to revolutionize image synthesis and classification by combining Deep Convolutional Generative Adversarial Networks (DCGANs) with Convolutional Neural Networks (CNNs). We seek to demonstrate the potential of DCGAN techniques in producing highly realistic images and achieving strong performance in image classification by utilizing these synthetic images during training.</p>
    </td>
  </tr> -->
  <tr>
    <td style="padding:2.5%;width:25%;vertical-align:middle;min-width:120px">
      <img src="../assets/images/projects/image-cap.png" alt="project image" style="width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="padding:2.5%;width:75%;vertical-align:middle">
      <h3>Automatic Image Annotation</h3>
      Yi Li, <strong>Weiyue Li</strong>, Linghang Kong, Yibo Wei, Shuangmu Wu
      <br>
      <a href="https://github.com/weiyueli7/Automatic-Image-Annotation/blob/main/report.pdf">report</a> / <a href="https://github.com/weiyueli7/Automatic-Image-Annotation">code</a>
      <p>In this project, we trained an algorithm to caption input images. This required the algorithm to identify objects in the images and match them to a corpus of text. We used PyTorch to implement multiple Recurrent Neural Network (RNN) models, including LSTM, Vanilla RNN, and a custom model (Architecture 2), to generate captions for the images in our dataset, specifically the well-known COCO Image Captioning Task.</p>
    </td>
  </tr>
  <tr>
    <td style="padding:2.5%;width:25%;vertical-align:middle;min-width:120px">
      <img src="../assets/images/projects/recommender.png" alt="project image" style="width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="padding:2.5%;width:75%;vertical-align:middle">
      <h3>Recipe Recommender System</h3>
      <strong>Weiyue Li</strong>, Yi Li, Xiaoyue Wang, Ruoyu Hou
      <br>
      <a href="https://github.com/weiyueli7/Recipe-Recommender/blob/main/report.pdf">report</a> / <a href="https://github.com/weiyueli7/Recipe-Recommender">code</a>
      <p>In this project, we first performed exploratory data analysis on datasets from food.com. We then implemented various types of recommendation system models to recommend recipes to users, predict ratings based on sentiment analysis, and predict recipe categories.</p>
    </td>
  </tr>

  <tr>
    <td style="padding:2.5%;width:25%;vertical-align:middle;min-width:120px">
      <img src="../assets/images/projects/mlp.png" alt="project image" style="width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="padding:2.5%;width:75%;vertical-align:middle">
      <h3>Optimization and Evaluation of Multi-layer Neural Networks: Exploring Regularization, Learning Rates, and Topologies</h3>
      <strong>Weiyue Li</strong>, Yi Li, Linghang Kong
      <br>
      <a href="https://github.com/weiyueli7/MLP/blob/main/report.pdf">report</a> / <a href="https://github.com/weiyueli7/MLP">code</a>
      <p>We implemented a multi-layer neural network equipped with forward and backward propagation, various regularization techniques, and momentum-based optimization. Our objective was to classify Japanese Hiragana handwritten characters from the KMNIST dataset, employing softmax as the output layer.</p>
    </td>
  </tr>


  <tr>
    <td style="padding:2.5%;width:25%;vertical-align:middle;min-width:120px">
      <img src="../assets/images/projects/svd.png" alt="project image" style="width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="padding:2.5%;width:75%;vertical-align:middle">
      <h3>Image Classification in Practice: High Efficiency and Performance From Singular Value Decomposition</h3>
      <strong>Weiyue Li</strong>
      <br>
      <a href="https://github.com/weiyueli7/Singular-Value-Decomposition-CV/blob/master/doc/report.pdf">report</a> / <a href="https://github.com/weiyueli7/Singular-Value-Decomposition-CV">code</a>
      <p>We implemented Logistic Regression with Stochastic Gradient Descent to classify Japanese Hiragana hand writings from the KMNIST dataset. We then used Singular Value Decomposition to reduce the size of images for the goals of decreasing memory allocations and hopefully increasing the performance of the model. After applying Singular Value Decomposition, we were able to achieve 99% of testing accuracy on classifying お and ま with 40% less memory allocation on the original images as well as around 87% of testing accuracy on classifying す and ま with 40% less memory allocation on the original images.</p>
    </td>
  </tr>


</table>


### Data Analysis

<table style="width:100%;border:0px;border-spacing:0px;border-collapse:separate;margin-right:auto;margin-left:auto;">
  <tr>
    <td style="padding:2.5%;width:25%;vertical-align:middle;min-width:120px">
      <img src="../assets/images/projects/covid.png" alt="project image" style="width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="padding:2.5%;width:75%;vertical-align:middle">
      <h3>Taming COVID-19 Statistics to Reflect Happiness Score Metrics</h3>
      Zirui Wang, <strong>Weiyue Li</strong>, Luning Yang, Yuru Zhou
      <br>
      <a href="https://zwcolin.github.io/COVID-19-Happiness-Score/FinalProjectGroup065-Fa21.html">report</a> / <a href="https://github.com/zwcolin/COVID-19-Happiness-Score">code</a>
      <p>In this project, we make use of the vaccination/death data by country, and explore the relationship between the set of {nation-wide onset date of vaccination, average new vaccination/death rate across different time spans}, and various metrics of happiness score in 2021. In particular, we are trying to figure out to what extent our independent varibles, namely all COVID-19 related data, are correlated to these metrics (i.e. social support, healthy life expectancy, perception of corruption, and generosity) of happiness scores.</p>
    </td>
  </tr>
</table>


### Data Visualization

<table style="width:100%;border:0px;border-spacing:0px;border-collapse:separate;margin-right:auto;margin-left:auto;">
  <tr>
    <td style="padding:2.5%;width:25%;vertical-align:middle;min-width:120px">
      <img src="../assets/images/projects/nfl-players.png" alt="project image" style="width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="padding:2.5%;width:75%;vertical-align:middle">
      <h3>NFL Players Visualization</h3>
      <strong>Weiyue Li</strong>
      <br>
      <a href="https://weiyueli7.github.io/NFL-Players-Viz/">report</a> / <a href="https://github.com/weiyueli7/NFL-Players-Viz">code</a>
      <p>In this project, I have developed an interactive dashboard aimed at providing high school varsity football players with a more comprehensive understanding of the critical considerations involved in becoming NFL players. In addition to drawing conclusions from past data, I have also made data-driven suggestions for young players. The primary objective of this project was to practice and enhance my skills in JavaScript, D3.js, and HTML.</p>
    </td>
  </tr>
</table>


### Econometrics

<table style="width:100%;border:0px;border-spacing:0px;border-collapse:separate;margin-right:auto;margin-left:auto;">
  <tr>
    <td style="padding:2.5%;width:25%;vertical-align:middle;min-width:120px">
      <img src="../assets/images/projects/wage-gap.png" alt="project image" style="width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="padding:2.5%;width:75%;vertical-align:middle">
      <h3>Occupation in Wage Gap Between Sex</h3>
      Jiahui Cai, <strong>Weiyue Li</strong>
      <br>
      <a href="https://github.com/weiyueli7/Wage-Gap-Analysis/blob/main/report.pdf">report</a> / <a href="https://github.com/weiyueli7/Wage-Gap-Analysis">code</a>
      <p>In this project, we perform a regression analysis to the IPUMS CPS data. In particular, we divide occupations into male-dominated, female-dominated, and equal-dominated and perform a regression analysis to aim for drawing a causal inference of occupation on the wage gap between sex during the COVID and post-COVID era.</p>
    </td>
  </tr>
</table>





### Others

This is a [website](https://sites.google.com/ucsd.edu/weiyue-li-first-year-japanese/home) that contains works I have done for the 3-quarter sequence of first-year Japanese courses I have taken in my freshman year of college. In case you are interested in taking the sequence, here are the topics you will work on.

This [page](/teaching_portfolio/) captures all of my work from EDS 124BR (Teach Computational Thinking). I think this course has helped me to become better at my job as a teaching assistant.







<!-- ### Selected Projects
{: .no_toc .text-delta }

1. TOC
{:toc}

#### Image Captioning

* Researched different hyperparameters on Long-Short-Term-Memory, Vanilla, and custom Recurrent Neural Networks on the COCO dataset to optimize image captioning.
* Achieved 66.7% BLEU-1 score and 7.69% BLEU-4 score on the testing dataset.

#### Image Classification

* Optimized Convolutional Neural Network on custom and pre-trained models to classify the Food101 dataset. Achieved 78% testing accuracy to classify 20 different classes of food images.
* Used regularization and momentum to optimize classifying handwritten Japanese Hiragana characters (KMNIST dataset) on back-propagation, logistic regression, and softmax regression. Achieved 87.25% testing accuracy on a multi-layer neural network with forward and backward propagation on 10 different classes; 98% testing accuracy on classifying two classes via logistic regression, and 70% testing accuracy on classifying 10 classes via softmax regression.

#### Taming COVID-19 Statistics to Reflect Happiness Score Metrics

* For the goal of promoting Covid-19 vaccinations to antivaccinists, researched how the onset of vaccination, the vaccination rates, and the death rate in a country are related to the happiness score of that country.
* Collected datasets on Covid-19 vaccination records, happiness scores by country, and population reports. Designed metrics and performed data cleaning and Exploratory Data Analysis to find meaningful patterns. Built statistical models and machine learning pipelines to further analyze the problem.
* Detected trends that countries have earlier access to Covid-19 vaccines and were vaccinating faster have higher happiness scores.


#### Party Affiliation of Congressmen by their Stock Trades
* Cleaned the dataset and imputed missing values with different statistical techniques. Performed Exploratory Data Analysis, Hypothesis tests, and Permutation tests to detect Congressmen's stock preferences on companies and industries from both parties.
* Extracted useful features from the dataset and built binary classification models to predict the party affiliation of Congressmen based on the information of their stock trades. Achieved 87.32% accuracy in determining the party affiliation after fine-tuning.


#### Crime Index and Wealth Analysis

* Collected and cleaned data of neighborhoods around San Diego from ArcGIS, and performed exploratory data analysis on average wealth in the communities and their corresponding crime indexes.
* Built data visualization and performed A\|B testings to better explain our variables, built machine learning pipelines for predictive analysis, and applied geospatial analysis to support our findings that areas with higher average household income are generally more robust to crimes. -->


