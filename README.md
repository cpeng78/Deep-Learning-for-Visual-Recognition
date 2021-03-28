# Deep-Learning-for-Visual-Recognition
 
This repository includes three assignments of CS7643 Deep Learning course at GeorgiaTech.

<div align=center><img src="assignment3/styles_images/starry_tubingen_before.png>
 <img src="assignment3/styles_images/starry_tubingen.png><div align=left>

The final project "strategy_evaluation" implemented a manual trading strategy and a machine learning based strategy learner. The manual strategy manually sets up trading rules based on 4 of the stock indicators implemented in the project "indicator_evaluation". The strategy learner automatically choose the trading actions by training a random forest learner and build the trading rules.

The trading strategies is backtested and compared with the market simulater implemented in project "marketsim". See [report](strategy_evaluation/report.pdf) for details.


## Projects: 
- **Assignment 1**: [Multi-layer Perceptron](assignment1). Deep Neural Networks are becoming more and more popular and widely applied to many ML-related domains. In this assignment, a simple pipeline of training neural networks was completed to recognize MNIST Handwritten Digits. Two newral network architectures, a simple softmax regression and a two-layer multi-layer perceptron, were implemented along with the code to load data, train and optimize these networks. See the [report1](assignment1/report-a1-cpeng78.pdf) for summarized experimental results and findings.

- **Assignment 2**: [Convolutional Neural Networks](assignment2). Convolutional Neural Networks (CNNs) are one of the major advancements in computer vision over the past decades. In this assignment, a simple CNN architecture is build from scratch. I also implement CNNs with the commonly used deep learning framework [Pytorch](https://pytorch.org/). Different experiments were run on imbalanced version of CIFAR-10 datasets to evaluate the model and techniques such as Class-Banlanced Focal Loss to deal with imbalanbced data. See the [report2](assignment2/report-a2-cpeng78.pdf) for details of the network design and experiment results.

- **Assignment 3**: [Network Visualization and Style Transfer](assignment3). In the Network Visualization part, we will explore the use of different type of attribution algorithms, both gradient and perturbation, for images, and understand their differences using the [Captum](https://captum.ai/) model interpretability tool for PyTorch. A Saliency Maps was implemented from scratch as an exercise. . See [report](assignment3/report-a3-cpeng78.pdf) for detailed results.

## Dependencies for Running Locally:

The projects are in Python (version 3.6), and rely heavily on a few important libraries. These libraries are under active development, which unfortunately means there can be some compatibility issues between versions.

To create an environment for these projects:
```
conda env create --file environment.yml
```
Activate the new environment:
```
conda activate ml4t
```
The list of each library and its version number provided in the conda environment format:
```
name: ml4t
dependencies:
- python=3.6
- cycler=0.10.0
- kiwisolver=1.1.0
- matplotlib=3.0.3
- numpy=1.16.3
- pandas=0.24.2
- pyparsing=2.4.0
- python-dateutil=2.8.0
- pytz=2019.1
- scipy=1.2.1
- seaborn=0.9.0
- six=1.12.0
- joblib=0.13.2
- pytest=5.0
- future=0.17.1
- pip
- pip:
  - pprofile==2.0.2
  - jsons==0.8.8
```
To test the code, you’ll need to set up your PYTHONPATH to include the grading module and the utility module util.py, which are both one directory up from the project directories. Here’s an example of how to run the grading script for the optional (deprecated) assignment Assess Portfolio (note, grade_anlysis.py is included in the template zip file for Assess Portfolio):
```
PYTHONPATH=../:. python grade_analysis.py
```
which assumes you’re typing from the folder ML4T_2020Fall/assess_portfolio/. This will print out a lot of information, and will also produce two text files: points.txt and comments.txt. It will probably be helpful to scan through all of the output printed out in order to trace errors to your code, while comments.txt will contain a succinct summary of which test cases failed and the specific errors (without the backtrace).
