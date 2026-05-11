# CNN Image Classification Lab - Full Theory and Oral Exam Notes

**Course:** Image and Video Technology and Analysis  
**Assignment:** 2nd individual laboratory exercise  
**Topic:** Convolutional Neural Networks (CNNs), CIFAR-100 classification, regularization, and transfer learning  

---

## Table of Contents

1. [What the Exercise Is About](#1-what-the-exercise-is-about)
2. [Dataset and Preprocessing](#2-dataset-and-preprocessing)
3. [Core CNN Theory](#3-core-cnn-theory)
4. [Paper 1 - LeNet](#4-paper-1---lenet)
5. [Paper 2 - AlexNet](#5-paper-2---alexnet)
6. [Paper 3 - VGG](#6-paper-3---vgg)
7. [Comparative Theory: LeNet vs AlexNet vs VGG](#7-comparative-theory-lenet-vs-alexnet-vs-vgg)
8. [Questions and Answers per Paper](#8-questions-and-answers-per-paper)
9. [Cross-Article Questions and Answers](#9-cross-article-questions-and-answers)
10. [CNN Architecture Concepts](#10-cnn-architecture-concepts)
11. [Dense Layers and Softmax](#11-dense-layers-and-softmax)
12. [MyCNN Architecture](#12-mycnn-architecture)
13. [Loss Functions and KL Divergence](#13-loss-functions-and-kl-divergence)
14. [Metrics: Accuracy, Precision, Recall, and F1-Score](#14-metrics-accuracy-precision-recall-and-f1-score)
15. [Question 1: Training CNNs from Scratch](#15-question-1-training-cnns-from-scratch)
16. [Question 2: Dropout, Data Augmentation, and Overfitting Control](#16-question-2-dropout-data-augmentation-and-overfitting-control)
17. [Question 3: Transfer Learning and Fine-Tuning](#17-question-3-transfer-learning-and-fine-tuning)
18. [Classification Head on Top of Pretrained Models](#18-classification-head-on-top-of-pretrained-models)
19. [High-Probability Oral Exam Questions](#19-high-probability-oral-exam-questions)
20. [Final Oral Exam Narrative](#20-final-oral-exam-narrative)

---

# 1. What the Exercise Is About

This laboratory exercise studies the theory and implementation of **Convolutional Neural Networks (CNNs)** for image classification.

The assignment has two main parts:

## Part A - Theoretical Part

The theoretical part studies three historically important CNN papers:

1. **Handwritten Digit Recognition with a Back-Propagation Network**  
   Associated with LeNet-style CNNs.

2. **ImageNet Classification with Deep Convolutional Neural Networks**  
   Associated with AlexNet.

3. **Very Deep Convolutional Networks for Large-Scale Image Recognition**  
   Associated with VGG.

The goal is to understand and compare:

- layers,
- filter sizes,
- activation functions,
- number of parameters,
- pooling methods,
- dropout,
- regularization,
- performance and design philosophy.

## Part B - Implementation Part

The implementation part uses a selected subset of **CIFAR-100**. The subset is determined by:

```python
team_seed = 26
```

The selected subset contains 20 classes:

```text
bus, chimpanzee, cloud, dinosaur, forest,
fox, girl, hamster, lobster, motorcycle,
oak_tree, pine_tree, seal, shrew, skunk,
snake, sweet_pepper, telephone, tiger, worm
```

The implementation has three main experimental questions:

1. **Question 1:** Implement and compare LeNet, AlexNet, VGG, and MyCNN.
2. **Question 2:** Improve MyCNN using dropout and data augmentation.
3. **Question 3:** Apply transfer learning using VGG19 and EfficientNetB0.

The central experimental question is:

> How do CNN architecture, model capacity, regularization, and pretrained representations affect image-classification performance on a 20-class CIFAR-100 subset?

---

# 2. Dataset and Preprocessing

## 2.1 CIFAR-100

CIFAR-100 contains:

- 100 fine-grained classes,
- 32 x 32 RGB images,
- 500 training images per class,
- 100 test images per class.

Each image has shape:

$$
32 \times 32 \times 3
$$

where:

- 32 is the image height,
- 32 is the image width,
- 3 is the number of color channels: red, green, blue.

## 2.2 Selected 20-Class Subset

The notebook uses only 20 selected classes, determined by `team_seed = 26`.

Since CIFAR-100 has 500 training images and 100 test images per class, the selected subset contains:

$$
20 \times 500 = 10000
$$

training/validation images, and:

$$
20 \times 100 = 2000
$$

test images.

A validation split of 15% is created from the selected training data. Therefore, the final split is:

| Split | Shape |
|---|---|
| Train | $(8500, 32, 32, 3)$ |
| Validation | $(1500, 32, 32, 3)$ |
| Test | $(2000, 32, 32, 3)$ |

## 2.3 Pixel Normalization

The original pixel values are in the range:

$$
[0,255]
$$

The notebook normalizes them to:

$$
[0,1]
$$

by dividing by 255.

This helps because neural networks train more smoothly when inputs are small and similarly scaled.

## 2.4 Label Remapping

This is a critical implementation detail.

Original CIFAR-100 labels are integers from:

$$
0,1,\ldots,99
$$

But after selecting only 20 classes, the model output layer is:

```python
Dense(20, activation="softmax")
```

Therefore, labels must be remapped to:

$$
0,1,\ldots,19
$$

If this remapping is not done, the model could receive a label such as 87 while only having 20 output neurons. That would be invalid.

### Oral Exam Answer

> The selected 20 CIFAR-100 classes still carry their original labels from the 0-99 CIFAR label space. Since our classifier outputs only 20 probabilities, the labels must be remapped to the range 0-19. This makes them compatible with a 20-neuron softmax output layer.

---

# 3. Core CNN Theory

## 3.1 Why CNNs Are Used for Images

Images have spatial structure:

- nearby pixels are related,
- local patterns such as edges and textures matter,
- the same feature can appear at different positions.

A fully connected network ignores this spatial structure. A CNN exploits it through:

1. local receptive fields,
2. weight sharing,
3. feature maps,
4. pooling,
5. hierarchical feature extraction.

The main CNN pipeline is:

$$
\text{image}
\rightarrow
\text{convolutional feature extraction}
\rightarrow
\text{compact representation}
\rightarrow
\text{classification}
$$

## 3.2 Convolution

A convolutional layer applies trainable filters over local regions of the input.

For an input with shape:

$$
H \times W \times C_{\text{in}}
$$

and a convolution with:

- kernel size $K \times K$,
- $C_{\text{out}}$ filters,

the number of parameters is:

$$
(K \cdot K \cdot C_{\text{in}} + 1) \cdot C_{\text{out}}
$$

The $+1$ is the bias term per filter, if biases are used.

## 3.3 Local Receptive Field

A **local receptive field** is the small region of the image that a convolutional filter sees at one position.

Example:

A $3 \times 3$ filter sees only a $3 \times 3$ patch at a time.

It then slides across the whole image.

### Why local receptive fields matter

They exploit the fact that useful visual patterns are local:

- edges,
- corners,
- color transitions,
- textures,
- small object parts.

### Simple Definition

> A local receptive field is the local patch of the image processed by a convolutional filter.

## 3.4 Weight Sharing

**Weight sharing** means that the same filter weights are used at every spatial location.

If a filter learns to detect a vertical edge, it can detect it anywhere in the image.

Benefits:

- fewer parameters,
- better generalization,
- translation-aware feature detection.

## 3.5 Feature Maps

A **feature map** is the result of applying one filter across the whole image.

If a convolutional layer has 32 filters, it produces 32 feature maps.

Conceptually:

```text
Filter/kernel = what feature is being searched for
Feature map = where that feature was found
```

## 3.6 Pooling

Pooling reduces spatial resolution.

Common types:

| Pooling Type | Meaning |
|---|---|
| Average pooling | takes the average in a local window |
| Max pooling | takes the maximum in a local window |

Pooling helps by:

- reducing computation,
- reducing spatial dimensions,
- increasing tolerance to small translations,
- increasing the effective receptive field of deeper layers.

## 3.7 Padding

Padding controls what happens at image boundaries.

| Padding | Effect |
|---|---|
| `valid` | no padding; spatial size shrinks |
| `same` | padding is added; spatial size is preserved for stride 1 |

For CIFAR images of only $32 \times 32$, using too many `valid` convolutions can shrink feature maps too quickly. Therefore, compact CNNs often use `padding="same"`.

## 3.8 Activation Functions

### Sigmoid and Tanh

Older networks such as LeNet used sigmoid/tanh-like nonlinearities.

Problems:

- saturation,
- small gradients,
- slower training in deeper networks.

### ReLU

AlexNet and VGG use ReLU:

$$
f(x)=\max(0,x)
$$

Benefits:

- faster training,
- less saturation for positive values,
- better for deeper networks.

---

# 4. Paper 1 - LeNet

## 4.1 Problem Addressed

The LeNet paper addresses handwritten digit recognition, especially digits from ZIP codes.

The key idea is that a neural network can learn directly from image pixels instead of requiring manually engineered features.

## 4.2 Main Contribution

LeNet introduced the core CNN principles:

- local receptive fields,
- shared weights,
- feature maps,
- average pooling/subsampling,
- end-to-end backpropagation.

## 4.3 Why Not a Fully Connected Network?

A fully connected network would connect every pixel to every neuron in the next layer.

This would cause:

- too many parameters,
- high computational cost,
- poor generalization,
- overfitting.

LeNet instead uses architectural constraints that match the structure of images.

## 4.4 Architecture Philosophy

LeNet is based on the idea:

$$
\text{local features}
\rightarrow
\text{subsampling}
\rightarrow
\text{higher-level features}
\rightarrow
\text{classification}
$$

The early layers detect simple local patterns. Later layers combine them into more abstract features.

## 4.5 Why LeNet Was Important

LeNet was important because it showed that:

- networks can learn directly from pixels,
- feature extraction can be learned,
- CNN architecture improves generalization,
- image-specific inductive biases are useful.

## 4.6 LeNet in the Notebook

In the notebook, LeNet acts as a baseline.

It is expected to be relatively weak on CIFAR-100 because:

- CIFAR images are more complex than handwritten digits,
- CIFAR has RGB natural images,
- LeNet has limited capacity,
- older activation and pooling choices are less powerful.

---

# 5. Paper 2 - AlexNet

## 5.1 Problem Addressed

AlexNet addresses large-scale natural image classification on ImageNet.

ImageNet contains:

- millions of images,
- 1000 classes in the ILSVRC challenge,
- high visual variability.

AlexNet showed that CNNs could outperform traditional handcrafted feature pipelines.

## 5.2 Main Contribution

AlexNet scaled CNNs to large image classification.

Its main innovations were:

- deeper CNN architecture,
- ReLU activations,
- GPU training,
- max pooling,
- Local Response Normalization,
- dropout,
- data augmentation.

## 5.3 ReLU

AlexNet used:

$$
f(x)=\max(0,x)
$$

instead of sigmoid or tanh.

ReLU helps because:

- it does not saturate for positive values,
- gradients propagate more effectively,
- training becomes much faster.

## 5.4 GPU Training

AlexNet was too large for practical CPU-only training at the time. It was trained on GPUs.

This was historically important because it showed the importance of combining:

$$
\text{large datasets}
+
\text{CNNs}
+
\text{GPU computation}
$$

## 5.5 Dropout

Dropout randomly sets activations to zero during training.

It reduces overfitting by preventing neurons from relying too much on each other.

In AlexNet, dropout was used mainly in the fully connected layers, which had many parameters.

## 5.6 Data Augmentation

AlexNet used:

- random crops,
- horizontal flips,
- RGB/color perturbations.

This increased the effective size and diversity of the training set.

## 5.7 Local Response Normalization

Local Response Normalization encourages competition between nearby feature maps.

It was useful in AlexNet, but later architectures such as VGG found it less necessary.

## 5.8 AlexNet in the Notebook

The notebook uses an adapted AlexNet-style model.

It preserves AlexNet principles:

- multiple convolutional layers,
- ReLU,
- max pooling,
- dropout,
- dense classifier.

But it is adapted to $32 \times 32$ CIFAR images, not copied exactly from the original ImageNet-scale model.

---

# 6. Paper 3 - VGG

## 6.1 Problem Addressed

VGG investigates how CNN depth affects classification performance.

The central question is:

> Does increasing CNN depth improve image classification accuracy?

The answer of the paper is yes, especially with 16-19 weight layers.

## 6.2 Main Contribution

VGG showed that very deep networks can be built using simple repeated blocks of small $3 \times 3$ convolutions.

## 6.3 Why $3 \times 3$ Filters?

A single $3 \times 3$ filter sees a small local region.

Stacking multiple $3 \times 3$ convolutions increases the effective receptive field.

Two $3 \times 3$ convolutions have an effective receptive field of:

$$
5 \times 5
$$

Three $3 \times 3$ convolutions have an effective receptive field of:

$$
7 \times 7
$$

This gives the effect of larger filters while adding more nonlinearities.

## 6.4 Parameter Advantage

Assume input and output both have $C$ channels.

A single $7 \times 7$ convolution has:

$$
49C^2
$$

weights.

Three $3 \times 3$ convolutions have:

$$
3 \cdot 9C^2 = 27C^2
$$

weights.

So stacked small filters can use fewer parameters and more nonlinear transformations.

## 6.5 VGG Block

A typical VGG block is:

```text
Conv 3x3 → ReLU → Conv 3x3 → ReLU → MaxPool
```

or deeper:

```text
Conv 3x3 → ReLU → Conv 3x3 → ReLU → Conv 3x3 → ReLU → MaxPool
```

## 6.6 VGG in the Notebook

The notebook uses a compact VGG-style model adapted to CIFAR.

It preserves:

- repeated $3 \times 3$ convolutional blocks,
- ReLU,
- max pooling,
- dense classifier,
- dropout.

It is not the full original VGG19, because full VGG19 is designed for $224 \times 224$ ImageNet images and is too large for the small CIFAR subset.

---

# 7. Comparative Theory: LeNet vs AlexNet vs VGG

## 7.1 General Comparison

| Characteristic | LeNet | AlexNet | VGG |
|---|---|---|---|
| Main task | Handwritten digit recognition | ImageNet classification | ImageNet classification |
| Input type | Small grayscale images | RGB natural images | RGB natural images |
| Main contribution | Core CNN principles | Large-scale deep CNN training | Depth with repeated $3 \times 3$ blocks |
| Main idea | Local features and weight sharing | Deep CNNs with GPUs, ReLU, dropout | Very deep uniform CNNs |

## 7.2 Network Structure

| Characteristic | LeNet | AlexNet | VGG |
|---|---|---|---|
| Typical depth | Small CNN | 8 learned layers | 16-19 learned layers |
| Convolutional layers | Usually 2 | 5 | 13-16 |
| Fully connected layers | Usually 3 | 3 | 3 |
| Parameters | tens of thousands in classical versions | about 60M | about 133M-144M |

## 7.3 Filters, Activations, and Pooling

| Characteristic | LeNet | AlexNet | VGG |
|---|---|---|---|
| Filter sizes | mainly $5 \times 5$ | $11 \times 11$, $5 \times 5$, $3 \times 3$ | mostly $3 \times 3$ |
| Activation | sigmoid/tanh-like | ReLU | ReLU |
| Pooling | average pooling/subsampling | max pooling | max pooling |
| Dropout | not original | yes | yes |

## 7.4 Design Evolution

The evolution can be summarized as:

$$
\text{LeNet: CNN principles}
\rightarrow
\text{AlexNet: large-scale deep CNNs}
\rightarrow
\text{VGG: depth and small-filter blocks}
$$

## 7.5 Key Difference: LeNet vs AlexNet

LeNet is small and designed for digit recognition. AlexNet is much larger and designed for natural RGB images.

AlexNet adds:

- ReLU,
- GPU training,
- dropout,
- data augmentation,
- much greater depth and capacity.

## 7.6 Key Difference: AlexNet vs VGG

AlexNet uses a more heterogeneous architecture with large early filters.

VGG uses a uniform architecture based almost entirely on $3 \times 3$ convolutions.

VGG focuses more systematically on depth.

---

# 8. Questions and Answers per Paper

## 8.1 LeNet Questions

### Q1. What is the main contribution of LeNet?

LeNet introduced the basic CNN principles: local receptive fields, weight sharing, feature maps, subsampling, and end-to-end backpropagation for image recognition.

### Q2. Why is LeNet not just a fully connected network?

Because its early layers use local connections and shared weights. This preserves image structure and reduces parameters.

### Q3. What is a local receptive field?

It is the small local image region processed by one filter or neuron.

### Q4. What is weight sharing?

The same filter is applied across all spatial locations.

### Q5. Why is subsampling used?

To reduce spatial resolution and increase tolerance to small translations and distortions.

### Q6. What prior knowledge does LeNet encode?

It encodes that images have local structure and that the same visual patterns can appear at different locations.

### Q7. Why was LeNet suitable for handwritten digits?

Digits are made of local patterns such as strokes, curves, and edges, and their class should not change under small translations.

### Q8. What is LeNet's limitation?

It has limited capacity and is less suitable for complex natural images.

---

## 8.2 AlexNet Questions

### Q1. What is the main contribution of AlexNet?

AlexNet showed that deep CNNs can achieve state-of-the-art performance on large-scale natural image classification.

### Q2. Why did AlexNet use ReLU?

ReLU trains faster than sigmoid/tanh because it reduces saturation and improves gradient flow.

### Q3. Why were GPUs important?

The model was large and trained on many images. GPUs made training practical.

### Q4. What is top-1 and top-5 error?

Top-1 error means the highest-probability prediction is wrong.  
Top-5 error means the true class is not among the five highest-probability predictions.

### Q5. Why did AlexNet use dropout?

To reduce overfitting, especially in large fully connected layers.

### Q6. Why did AlexNet use data augmentation?

To increase effective training data diversity and improve generalization.

### Q7. What is Local Response Normalization?

It is a normalization method that creates competition between neighboring feature maps.

### Q8. Why did AlexNet use large filters at the beginning?

Because ImageNet images are large, and large early filters with stride quickly capture broad patterns and reduce spatial dimensions.

### Q9. What is a weakness of AlexNet?

It has large fully connected layers and a high parameter count, making it prone to overfitting.

---

## 8.3 VGG Questions

### Q1. What is the main contribution of VGG?

VGG showed that increasing CNN depth using repeated small $3 \times 3$ filters improves image classification performance.

### Q2. Why does VGG use $3 \times 3$ filters?

Because stacking small filters increases receptive field while adding nonlinearities and reducing parameters compared with large filters.

### Q3. What is effective receptive field?

It is the region of the original input that affects a later-layer activation.

### Q4. Why is VGG considered simple?

Because it uses repeated blocks of convolution, ReLU, and max pooling without complicated modules.

### Q5. Why does VGG have many parameters?

Mostly because of its large fully connected layers.

### Q6. Why is VGG good for transfer learning?

Because it learns general visual features that transfer well to other computer vision tasks.

### Q7. Why did VGG not rely on LRN?

Because LRN did not significantly improve performance but increased memory and computation.

### Q8. What is VGG's main weakness?

It is computationally heavy and has many parameters.

---

# 9. Cross-Article Questions and Answers

## Q1. What is the historical evolution from LeNet to AlexNet to VGG?

LeNet introduced the CNN principles. AlexNet scaled CNNs to large natural-image classification using GPUs and modern training techniques. VGG showed that deeper networks with small filters produce stronger representations.

## Q2. What do all three models have in common?

All use convolutional feature extraction, spatial hierarchy, and a final classifier.

## Q3. How does feature extraction evolve across the three papers?

- LeNet: simple local digit features,
- AlexNet: richer natural image features,
- VGG: deeper hierarchical visual representations.

## Q4. Why does depth help image recognition?

Early layers learn edges and colors. Middle layers learn textures and object parts. Deeper layers learn more semantic patterns.

## Q5. Why not simply use the largest model?

Because small datasets can cause large models to overfit. Capacity must match available data.

## Q6. Why use transfer learning?

Pretrained models already learned useful visual features from large datasets, so they require less data to perform well on a new task.

## Q7. Why are CNNs better than MLPs for images?

CNNs exploit locality and weight sharing. MLPs ignore spatial structure and require many more parameters.

## Q8. What is the relation between pooling and invariance?

Pooling provides limited tolerance to small translations by summarizing local regions.

## Q9. What is the main message of the three papers together?

Successful image classification depends on:

$$
\text{locality}
+
\text{weight sharing}
+
\text{depth}
+
\text{nonlinearity}
+
\text{regularization}
+
\text{data}
$$

---

# 10. CNN Architecture Concepts

## 10.1 What Is a Layer?

A layer is one processing stage in a neural network.

Common CNN layers:

| Layer | Role |
|---|---|
| `Conv2D` | extracts local features |
| `ReLU` | adds nonlinearity |
| `MaxPooling2D` | downsamples feature maps |
| `BatchNormalization` | stabilizes training |
| `Dropout` | reduces overfitting |
| `Flatten` | converts feature maps to a vector |
| `GlobalAveragePooling2D` | summarizes each feature map |
| `Dense` | fully connected classification |
| `Softmax` | outputs class probabilities |

## 10.2 What Is a Block?

A block is a repeated group of layers.

Example simple block:

```text
Conv2D → ReLU → MaxPooling
```

Example VGG block:

```text
Conv2D → ReLU → Conv2D → ReLU → MaxPooling
```

Blocks are used to organize architectures.

## 10.3 Feature Extractor vs Classifier

A CNN usually has two parts:

```text
Feature extractor → Classifier
```

The feature extractor contains convolutional and pooling layers.

The classifier contains dense layers and the final softmax output.

## 10.4 How Convolutional Layers Connect to Fully Connected Layers

Convolutional layers output a 3D tensor:

$$
H \times W \times C
$$

Before dense layers, this must be converted to a vector.

There are two common methods:

### Flatten

$$
H \times W \times C \rightarrow HWC
$$

This can create many parameters.

### GlobalAveragePooling2D

$$
H \times W \times C \rightarrow C
$$

This produces one value per feature channel and greatly reduces parameters.

---

# 11. Dense Layers and Softmax

## 11.1 Dense Layer

A Dense layer is a fully connected layer.

Each neuron is connected to all outputs of the previous layer.

Mathematically:

$$
z = Wx + b
$$

where:

- $x$ is the input vector,
- $W$ is the weight matrix,
- $b$ is the bias,
- $z$ is the output vector.

Example:

```python
Dense(128, activation="relu")
```

means:

- 128 neurons,
- fully connected to previous vector,
- ReLU activation.

## 11.2 Role of Dense Layers in CNNs

Convolutional layers extract features.

Dense layers combine those features to make a final class decision.

Example:

```text
Convolutional layers:
"edges, textures, shapes were found"

Dense layers:
"combine these features to decide the class"
```

## 11.3 Softmax

Softmax converts raw scores into probabilities.

For 20 classes:

```python
Dense(20, activation="softmax")
```

outputs:

$$
[p_1,p_2,\ldots,p_{20}]
$$

where:

$$
0 \le p_i \le 1
$$

and:

$$
\sum_{i=1}^{20} p_i = 1
$$

Softmax formula:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{20} e^{z_j}}
$$

The predicted class is:

$$
\hat{y} = \arg\max_i p_i
$$

## 11.4 Oral Exam Answer

> A Dense layer is a fully connected layer. It combines the features extracted by convolutional layers. The final Dense layer has 20 neurons because we have 20 classes. Softmax converts these 20 raw scores into probabilities that sum to 1, and the predicted class is the one with the highest probability.

---

# 12. MyCNN Architecture

## 12.1 Purpose of MyCNN

MyCNN is the custom CNN architecture designed specifically for the selected 20-class CIFAR-100 subset.

It aims to be:

- stronger than LeNet,
- more compact than AlexNet/VGG,
- suitable for $32 \times 32$ images,
- less prone to overfitting.

## 12.2 High-Level Structure

```text
Input image
→ Convolutional feature extractor
→ GlobalAveragePooling2D
→ Dense classifier
→ Softmax output
```

Actual architecture:

```text
Input: 32 × 32 × 3

Block 1:
Conv-BN-ReLU, 32 filters
Conv-BN-ReLU, 32 filters
MaxPooling

Block 2:
Conv-BN-ReLU, 64 filters
Conv-BN-ReLU, 64 filters
MaxPooling

Block 3:
Conv-BN-ReLU, 128 filters
Conv-BN-ReLU, 128 filters

Classifier:
GlobalAveragePooling2D
Dense(128, ReLU)
Dense(20, Softmax)
```

## 12.3 Conv-BN-ReLU Unit

Each unit is:

```text
Conv2D → BatchNormalization → ReLU
```

### Conv2D

Learns local filters.

### BatchNormalization

Stabilizes activations and improves training.

### ReLU

Adds nonlinearity.

## 12.4 Why $3 \times 3$ Convolutions?

$3 \times 3$ filters capture local patterns while keeping parameters manageable.

Stacking several $3 \times 3$ convolutions creates larger effective receptive fields, following VGG's design philosophy.

## 12.5 Why `padding="same"`?

Because CIFAR images are small. `padding="same"` preserves spatial size after convolution and prevents feature maps from shrinking too quickly.

## 12.6 Why `use_bias=False`?

Because each convolution is followed by Batch Normalization. BatchNorm already has trainable shift parameters, so the convolutional bias is unnecessary.

## 12.7 Shape Flow

| Stage | Output Shape |
|---|---|
| Input | $32 \times 32 \times 3$ |
| Block 1 Conv | $32 \times 32 \times 32$ |
| Block 1 Pool | $16 \times 16 \times 32$ |
| Block 2 Conv | $16 \times 16 \times 64$ |
| Block 2 Pool | $8 \times 8 \times 64$ |
| Block 3 Conv | $8 \times 8 \times 128$ |
| GlobalAveragePooling2D | $128$ |
| Dense | $128$ |
| Output | $20$ |

## 12.8 Why Channels Increase

Spatial dimensions decrease:

$$
32 \times 32
\rightarrow
16 \times 16
\rightarrow
8 \times 8
$$

Channels increase:

$$
32
\rightarrow
64
\rightarrow
128
$$

This is common in CNNs: lower spatial resolution, richer feature representation.

## 12.9 GlobalAveragePooling2D

The final feature tensor is:

$$
8 \times 8 \times 128
$$

GlobalAveragePooling2D converts it to:

$$
128
$$

by averaging each channel spatially.

For each channel:

$$
z_c =
\frac{1}{8 \cdot 8}
\sum_{i=1}^{8}
\sum_{j=1}^{8}
x_{i,j,c}
$$

## 12.10 Why GlobalAveragePooling Helps

If we used Flatten:

$$
8 \times 8 \times 128 = 8192
$$

A Dense(128) after that would have over one million parameters.

With GlobalAveragePooling, Dense(128) has only:

$$
128 \cdot 128 + 128 = 16512
$$

parameters.

This reduces overfitting.

## 12.11 MyCNN Oral Exam Answer

> MyCNN is a custom CNN designed for $32 \times 32$ CIFAR images. It has three convolutional blocks using Conv-BatchNorm-ReLU units with $3 \times 3$ filters. The number of filters increases from 32 to 64 to 128, while max pooling reduces spatial size from $32 \times 32$ to $16 \times 16$ and then $8 \times 8$. After the convolutional feature extractor, GlobalAveragePooling2D converts the $8 \times 8 \times 128$ tensor into a 128-dimensional vector. Then a Dense(128, ReLU) layer combines the features and Dense(20, softmax) outputs one probability per selected CIFAR-100 class. This design balances capacity and generalization better than the heavier adapted models.

---

# 13. Loss Functions and KL Divergence

## 13.1 Classification Loss

For multi-class classification, common losses include:

- categorical cross-entropy,
- sparse categorical cross-entropy,
- KL divergence.

The correct label format matters.

| Loss | Label Format |
|---|---|
| SparseCategoricalCrossentropy | integer labels |
| CategoricalCrossentropy | one-hot labels |
| KLDivergence | one-hot probability distributions |

## 13.2 One-Hot Labels

If the true class is class 3 in a 20-class problem, the one-hot vector is:

$$
y = [0,0,1,0,\ldots,0]
$$

Only the correct class is 1.

## 13.3 Softmax Output

The model outputs:

$$
\hat{y} = [\hat{y}_1,\hat{y}_2,\ldots,\hat{y}_{20}]
$$

where:

$$
\sum_{i=1}^{20} \hat{y}_i = 1
$$

Thus the output is a probability distribution.

## 13.4 KL Divergence

KL divergence measures how different one probability distribution is from another:

$$
D_{KL}(y \parallel \hat{y})
=
\sum_{i=1}^{20}
y_i \log \left(\frac{y_i}{\hat{y}_i}\right)
$$

Because $y$ is one-hot, all terms where $y_i=0$ vanish.

If the correct class is $k$:

$$
y_k = 1
$$

then:

$$
D_{KL}(y \parallel \hat{y})
=
\log\left(\frac{1}{\hat{y}_k}\right)
=
-\log(\hat{y}_k)
$$

## 13.5 Categorical Cross-Entropy

Categorical cross-entropy is:

$$
CE(y,\hat{y})
=
-\sum_{i=1}^{20} y_i \log(\hat{y}_i)
$$

For one-hot labels, only the correct class contributes:

$$
CE(y,\hat{y})
=
-\log(\hat{y}_k)
$$

Therefore, for one-hot labels:

$$
D_{KL}(y \parallel \hat{y})
=
CE(y,\hat{y})
$$

## 13.6 Simple Example

If the correct class is tiger and the model predicts:

```text
tiger: 0.80
bus:   0.05
cloud: 0.10
snake: 0.05
```

the loss is:

$$
-\log(0.80)
$$

If the model predicts:

```text
tiger: 0.05
bus:   0.70
cloud: 0.15
snake: 0.10
```

the loss is:

$$
-\log(0.05)
$$

which is much larger.

## 13.7 Oral Exam Answer

> KL divergence is valid here because both the target and prediction are probability distributions. The targets are one-hot encoded, and the output uses softmax. For a one-hot target, all KL terms are zero except the true class term, so KL divergence becomes $-\log(\hat{y}_{true})$, which is the same expression as categorical cross-entropy for one-hot labels.

---

# 14. Metrics: Accuracy, Precision, Recall, and F1-Score

## 14.1 Accuracy

Accuracy is:

$$
\text{Accuracy}
=
\frac{\text{correct predictions}}{\text{total predictions}}
$$

It is simple, but it may hide weak performance on individual classes.

## 14.2 Precision

For one class:

$$
\text{Precision}
=
\frac{TP}{TP+FP}
$$

Precision answers:

> When the model predicts this class, how often is it correct?

## 14.3 Recall

For one class:

$$
\text{Recall}
=
\frac{TP}{TP+FN}
$$

Recall answers:

> Of all true examples of this class, how many did the model find?

## 14.4 F1-Score

F1 combines precision and recall:

$$
F_1 =
2 \cdot
\frac{\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

## 14.5 Macro-F1

Macro-F1 computes F1 per class and averages equally:

$$
F_{1,\text{macro}}
=
\frac{1}{C}
\sum_{c=1}^{C} F_{1,c}
$$

where:

$$
C = 20
$$

Macro-F1 is useful because it treats all classes equally.

## 14.6 Weighted-F1

Weighted-F1 weights each class by the number of examples:

$$
F_{1,\text{weighted}}
=
\sum_{c=1}^{C}
\frac{n_c}{N}F_{1,c}
$$

If all classes have equal support, macro-F1 and weighted-F1 are very close or identical.

## 14.7 Why Use F1 Instead of Only Accuracy?

F1 gives more information about class-wise behavior.

It is useful when:

- class imbalance exists,
- some classes are harder than others,
- we want a balanced evaluation across classes.

## 14.8 Why Compute F1 with Full Predictions?

Macro-F1 is not naturally decomposable over mini-batches. Therefore, it is better to compute it after predicting on the full train/validation/test set.

### Oral Exam Answer

> We compute macro-F1 over the full dataset because F1 depends on global true positives, false positives, and false negatives. Batch-wise F1 can be misleading, while full-set F1 gives correct class-balanced evaluation.

---

# 15. Question 1: Training CNNs from Scratch

## 15.1 Models Trained

Question 1 trains:

- LeNet,
- AlexNet,
- VGG,
- MyCNN.

## 15.2 Fair Comparison

All models are trained under the same selected setup:

- same optimizer,
- same loss,
- same batch size,
- same number of epochs,
- same evaluation metric.

This ensures that architectural differences are the main factor in performance.

## 15.3 Why Adapt Original Architectures?

Original AlexNet and VGG were designed for ImageNet images:

$$
224 \times 224 \times 3
$$

CIFAR images are:

$$
32 \times 32 \times 3
$$

Directly copying original ImageNet architectures would be inappropriate because:

- spatial dimensions would collapse too fast,
- parameter counts would be too large,
- overfitting would increase,
- computation would be unnecessary.

Therefore, the notebook preserves the architectural principles, not the exact original dimensions.

## 15.4 Common Training Configuration

The notebook selected a common configuration after a grid search:

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Kullback-Leibler Divergence |
| Batch size | 32 |
| Final epochs | 50 |

## 15.5 Interpretation of Model Performance

Expected general behavior:

| Model | Expected Behavior |
|---|---|
| LeNet | simple baseline, likely underfits |
| AlexNet | higher capacity, but may overfit |
| VGG | stronger hierarchy, but still parameter-heavy |
| MyCNN | best CIFAR-specific balance |

## 15.6 Why MyCNN Performs Well

MyCNN is specifically designed for CIFAR-sized images.

It uses:

- $3 \times 3$ convolutions,
- Batch Normalization,
- ReLU,
- MaxPooling,
- GlobalAveragePooling,
- moderate dense classifier.

This gives a good balance between capacity and generalization.

---

# 16. Question 2: Dropout, Data Augmentation, and Overfitting Control

## 16.1 Goal

Question 2 focuses only on MyCNN and tries to reduce overfitting using:

- dropout,
- data augmentation,
- dropout + data augmentation.

## 16.2 Overfitting

Overfitting happens when:

- training F1 is high,
- validation F1 is significantly lower.

This means the model memorizes the training data instead of learning general patterns.

## 16.3 Dropout

Dropout randomly sets activations to zero during training.

If dropout rate is $p$, each selected activation is dropped with probability:

$$
p
$$

Benefits:

- reduces co-adaptation,
- improves robustness,
- reduces overfitting.

Too much dropout can cause underfitting.

## 16.4 Where Dropout Is Applied

In MyCNN, dropout is applied after the dense layer:

```text
Dense(128, ReLU)
→ Dropout
→ Dense(20, Softmax)
```

It is not applied aggressively inside every convolutional block.

Reason:

- convolutional feature extraction should remain stable,
- BatchNorm already helps regularize,
- dense classifier is more likely to overfit.

## 16.5 Data Augmentation

Data augmentation creates transformed versions of training images without changing labels.

Examples:

- random horizontal flip,
- random translation,
- random rotation,
- random zoom,
- random contrast.

It improves generalization by exposing the model to more variations.

## 16.6 Why Augmentation Is Applied at the Input

Augmentation modifies the image itself.

Example:

```text
tiger image
→ shifted tiger
→ rotated tiger
→ flipped tiger
```

The label remains `tiger`.

## 16.7 Interpreting Q2

The best model is not necessarily the one with the highest training score.

A good model should have:

- high validation/test F1,
- small train-validation gap,
- stable learning curves.

## 16.8 Oral Exam Answer

> In Question 2, we regularized MyCNN with dropout and data augmentation. Dropout prevents the dense classifier from relying too much on specific neurons, while augmentation creates label-preserving transformations of training images. The goal is not only to slightly improve test F1, but also to reduce the train-validation gap and improve generalization.

---

# 17. Question 3: Transfer Learning and Fine-Tuning

## 17.1 Goal

Question 3 uses pretrained CNNs:

- VGG19,
- EfficientNetB0.

These were pretrained on ImageNet and then adapted to the 20-class CIFAR subset.

## 17.2 Transfer Learning

Transfer learning reuses features learned from a large source dataset.

Early CNN layers learn general features such as:

- edges,
- colors,
- textures.

Deeper layers learn more semantic features.

These learned representations can help on a new dataset, especially when the new dataset is small.

## 17.3 Why Pretrained Models Help

Training from scratch on 8500 images is limited.

Pretrained models have already learned from millions of images. Therefore, they provide strong visual features before seeing our dataset.

## 17.4 Frozen-Base Training

In the first stage:

```python
base_model.trainable = False
```

The pretrained base is frozen.

Only the new classification head is trained.

Why?

- avoids damaging pretrained features,
- trains faster,
- stabilizes the new classifier.

## 17.5 Fine-Tuning

In the second stage, some top layers of the pretrained base are unfrozen.

The model is then trained with a small learning rate.

Why small learning rate?

Because we only want to slightly adapt pretrained weights, not overwrite them.

## 17.6 Why Not Fine-Tune Everything Immediately?

The new classification head starts randomly initialized. If the entire pretrained network is trainable from the beginning, large gradients can damage useful pretrained features.

Therefore:

1. train head first,
2. fine-tune top layers carefully.

## 17.7 Why Keep BatchNorm Frozen in EfficientNet?

BatchNorm layers store moving mean and variance statistics.

With a small dataset, updating them can destabilize the pretrained representation.

Therefore, BatchNorm layers are often kept frozen during fine-tuning.

## 17.8 Preprocessing for Transfer Learning

CIFAR images are $32 \times 32$.

Pretrained models expect larger images. The notebook resizes to:

$$
128 \times 128 \times 3
$$

Then applies model-specific preprocessing.

Important:

If CIFAR images were normalized to $[0,1]$, they may need to be multiplied back by 255 before official Keras preprocessing.

## 17.9 Why EfficientNetB0 Can Outperform VGG19

EfficientNetB0 is more modern and parameter-efficient.

It uses a better scaling strategy and stronger feature extraction for fewer parameters.

VGG19 is powerful but older and heavier.

## 17.10 Oral Exam Answer

> Transfer learning works well because VGG19 and EfficientNetB0 have already learned useful visual features from ImageNet. We remove their original classifier, freeze the convolutional base, and train a new 20-class head. Then we fine-tune the top layers with a small learning rate. EfficientNetB0 usually performs better because it is more modern and parameter-efficient than VGG19.

---

# 18. Classification Head on Top of Pretrained Models

## 18.1 Why Add a New Head?

VGG19 and EfficientNetB0 were originally trained for ImageNet:

$$
1000
$$

classes.

Our task has:

$$
20
$$

classes.

Therefore, we remove the original classifier by using:

```python
include_top=False
```

and add a custom classification head.

## 18.2 High-Level Structure

```text
Input image
→ Pretrained convolutional base
→ Custom classification head
→ 20-class softmax output
```

## 18.3 Actual Head Architecture

```text
GlobalAveragePooling2D
Dense(256, ReLU)
Dropout(0.4)
Dense(20, Softmax)
```

## 18.4 GlobalAveragePooling2D

The pretrained base outputs a feature tensor:

$$
H \times W \times C
$$

GlobalAveragePooling2D converts it to:

$$
C
$$

by averaging each channel.

Formula:

$$
z_c =
\frac{1}{H \cdot W}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
x_{i,j,c}
$$

Why use it?

- avoids huge Flatten vectors,
- reduces trainable parameters,
- reduces overfitting.

## 18.5 Dense(256, ReLU)

This layer learns task-specific combinations of pretrained features.

It adapts general ImageNet visual features to our selected CIFAR classes.

## 18.6 Dropout(0.4)

This regularizes the classification head by randomly dropping 40% of activations during training.

It reduces overfitting in the newly trained classifier.

## 18.7 Dense(20, Softmax)

The final layer outputs one probability per selected CIFAR-100 class.

It has 20 neurons because there are 20 classes.

## 18.8 Frozen vs Fine-Tuned Stage

### Frozen Stage

```text
Pretrained base: frozen
Classification head: trainable
```

Only the head learns.

### Fine-Tuned Stage

```text
Top pretrained layers: trainable
Classification head: trainable
```

The model adapts high-level pretrained features.

## 18.9 Oral Exam Answer

> We used `include_top=False` to remove the original ImageNet classifier. Then we added a custom head: GlobalAveragePooling2D, Dense(256, ReLU), Dropout(0.4), and Dense(20, softmax). Global average pooling converts feature maps into a compact vector. The dense layer learns task-specific combinations of features. Dropout reduces overfitting, and the final softmax outputs probabilities for the 20 selected CIFAR-100 classes.

---

# 19. High-Probability Oral Exam Questions

## Q1. What is the purpose of the exercise?

The exercise studies CNNs theoretically and experimentally. It compares LeNet, AlexNet, and VGG, trains adapted CNNs on a CIFAR-100 subset, regularizes MyCNN, and applies transfer learning with VGG19 and EfficientNetB0.

## Q2. Why are labels remapped?

Because the selected classes have original CIFAR-100 labels from 0 to 99, but the model outputs only 20 probabilities. Labels must be remapped to 0-19.

## Q3. Why use CNNs instead of fully connected networks?

CNNs exploit image locality and weight sharing, reducing parameters and improving generalization.

## Q4. What is a local receptive field?

It is the small image patch seen by a convolutional filter at one position.

## Q5. What is weight sharing?

The same filter is reused across all spatial positions.

## Q6. What is a feature map?

The output map produced by applying one filter over the image.

## Q7. What is a block?

A block is a repeated group of layers, such as Conv-ReLU-Conv-ReLU-Pool.

## Q8. Why are convolutional layers followed by dense layers?

Convolutional layers extract visual features. Dense layers combine those features for classification.

## Q9. What does softmax do?

It converts raw class scores into probabilities that sum to one.

## Q10. What is LeNet's main contribution?

It introduced core CNN principles for image recognition.

## Q11. What is AlexNet's main contribution?

It scaled CNNs to large-scale ImageNet classification using ReLU, GPUs, dropout, and augmentation.

## Q12. What is VGG's main contribution?

It showed the importance of depth using repeated $3 \times 3$ convolutional blocks.

## Q13. Why use $3 \times 3$ filters in VGG?

Stacked $3 \times 3$ filters increase the receptive field while adding more nonlinearities and controlling parameters.

## Q14. Why use GlobalAveragePooling in MyCNN?

It reduces the feature tensor to one value per channel, avoiding a huge dense classifier and reducing overfitting.

## Q15. Why did MyCNN outperform heavier models from scratch?

Because it was designed specifically for CIFAR-sized images and balances capacity with regularization.

## Q16. What is dropout?

Dropout randomly sets activations to zero during training to reduce overfitting.

## Q17. What is data augmentation?

It creates label-preserving transformations of training images to improve generalization.

## Q18. Why use transfer learning?

Pretrained models already contain useful visual features learned from ImageNet.

## Q19. Why freeze the pretrained base first?

To train the new classifier head without damaging pretrained features.

## Q20. Why fine-tune later?

To adapt high-level pretrained features to the new dataset.

## Q21. Why use a small learning rate during fine-tuning?

To avoid overwriting useful pretrained weights.

## Q22. Why is KL divergence valid here?

Because the labels are one-hot encoded and the output is a softmax distribution. For one-hot labels, KL divergence becomes $-\log(\hat{y}_{true})$, equivalent to categorical cross-entropy.

## Q23. Why use macro-F1?

Macro-F1 gives equal importance to each class and is useful for evaluating class-wise performance.

## Q24. What does a large train-validation F1 gap mean?

It indicates overfitting.

## Q25. Why can EfficientNetB0 outperform VGG19?

EfficientNetB0 is more modern and parameter-efficient, usually giving better generalization.

---

# 20. Final Oral Exam Narrative

Use this if asked to summarize the whole notebook:

> The notebook studies CNN image classification both theoretically and experimentally. In Part A, I compare LeNet, AlexNet, and VGG. LeNet introduces the basic CNN ideas: local receptive fields, weight sharing, feature maps, and subsampling. AlexNet scales CNNs to ImageNet using ReLU, GPUs, dropout, and data augmentation. VGG shows that depth and repeated $3 \times 3$ convolutional blocks improve visual representations.
>
> In Part B, I use a 20-class CIFAR-100 subset selected by `team_seed = 26`. The images are normalized and the original CIFAR labels are remapped to 0-19 so that they match the 20-neuron softmax output. In Question 1, I train adapted LeNet, AlexNet, VGG, and MyCNN models under the same setup. MyCNN performs best from scratch because it is designed for $32 \times 32$ images and uses Conv-BatchNorm-ReLU blocks, max pooling, global average pooling, and a compact classifier.
>
> In Question 2, I improve MyCNN using dropout and data augmentation to reduce overfitting. Dropout regularizes the dense classifier, while augmentation exposes the model to transformed training images. The best regularized MyCNN reduces the train-validation gap and improves generalization.
>
> In Question 3, I apply transfer learning with VGG19 and EfficientNetB0. I remove the original ImageNet classifier with `include_top=False`, add a new classification head with GlobalAveragePooling2D, Dense(256, ReLU), Dropout(0.4), and Dense(20, softmax), first train the head with the base frozen, and then fine-tune the top layers with a small learning rate. Transfer learning performs best overall because pretrained ImageNet models already contain strong visual features, and EfficientNetB0 performs especially well due to its parameter-efficient architecture.
