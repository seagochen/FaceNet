@[toc]

# Authors and Publishment

## Authors
* Florian Schroff / Google Inc.
* Dmitry Kalenichenko / Google Inc.
* James Philbin / Google Inc.


## Bibtex
Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering, Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 815-823.

## Categories
Computer Graphics, Deep Learning, Object Detection


# 0.  Abstract
Despite significant recent advances in the field of face recognition [10, 14, 15, 17], implementing face verification and recognition efficiently at scale presents serious challenges to current approaches. In this paper we present a system, called FaceNet, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors.

> 尽管人脸识别领域最近取得了重大进展（如[10,14,15,17]），但对于当前方法而言，在大规模上高效地实现人脸验证和识别仍然是一个严峻的挑战。本文提出了一个名为FaceNet的系统，它直接学习从人脸图像映射到紧凑的欧几里得空间的映射，其中距离直接对应于人脸相似性的测量。一旦生成了这个空间，诸如人脸识别、验证和聚类等任务就可以使用FaceNet嵌入作为特征向量，使用标准技术轻松实现。

Our method uses a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches. To train, we use triplets of roughly aligned matching / non-matching face patches generated using a novel online triplet mining method. The benefit of our approach is much greater representational efficiency: we achieve state-of-the-art face recognition performance using only 128-bytes per face.

> 我们的方法使用深度卷积网络，直接优化嵌入本身，而不是先前的深度学习方法中的中间瓶颈层。为了训练，我们使用通过一种新颖的在线三元组挖掘方法生成的大致对齐的匹配/不匹配的人脸补丁三元组。我们的方法的优势在于更高的表示效率：我们仅使用128字节每张人脸，就实现了最先进的人脸识别性能。

On the widely used Labeled Faces in the Wild (LFW) dataset, our system achieves a new record accuracy of 99.63%. On YouTube Faces DB it achieves 95.12%. Our system cuts the error rate in comparison to the best published result [15] by 30% on both datasets.

> 在广泛使用的Labeled Faces in the Wild（LFW）数据集上，我们的系统实现了99.63％的新记录精度。在YouTube Faces DB上，它达到了95.12％。与最佳发布结果[15]相比，我们的系统在两个数据集上将错误率降低了30％。


# 1. Introduction
In this paper we present a unified system for face verification (is this the same person), recognition (who is this person) and clustering (find common people among these faces). Our method is based on learning a Euclidean embedding per image using a deep convolutional network. The network is trained such that the squared L2 distances in the embedding space directly correspond to face similarity: faces of the same person have small distances and faces of distinct people have large distances.

> 在本文中，我们提出了一个联合系统，用于人脸验证（是否为同一人）、识别（此人是谁）和聚类（在这些人脸中找到共同的人）。我们的方法基于使用深度卷积网络学习每个图像的欧几里得嵌入。训练网络的方式是，嵌入空间中的平方L2距离直接对应于人脸相似性：同一人的脸具有小距离，不同人的脸具有大距离。

Once this embedding has been produced, then the aforementioned tasks become straight-forward: face verification simply involves thresholding the distance between the two embeddings; recognition becomes a KNN classification problem; and clustering can be achieved using off-the- shelf techniques such as k-means or agglomerative clustering.

> 一旦产生了这个嵌入，那么上述任务就变得简单明了：人脸验证仅涉及阈值两个嵌入之间的距离；识别变成了KNN分类问题；可以使用现成技术（例如k-means或凝聚聚类）实现聚类。


![在这里插入图片描述](https://img-blog.csdnimg.cn/3307502b5a5e42af82a1e3a975998647.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)*Figure 1. Illumination and Pose invariance. Pose and illumination have been a long standing problem in face recognition. This figure shows the output distances of FaceNet between pairs of faces of the same and a different person in different pose and illumination combinations. A distance of 0.0 means the faces are identical, 4.0 corresponds to the opposite spectrum, two different identities. You can see that a threshold of 1.1 would classify every pair correctly.*

> 图1. 光照和姿势不变性。姿势和光照是人脸识别中一个长期存在的问题。该图显示了FaceNet在不同姿势和光照组合下同一人和不同人的面孔对之间的输出距离。0.0的距离意味着这两张脸是相同的，4.0对应的是相反的光谱，即两个不同的身份。你可以看到，1.1的阈值可以正确地对每一对进行分类。

Previous face recognition approaches based on deep networks use a classification layer [15, 17] trained over a set of known face identities and then take an intermediate bottleneck layer as a representation used to generalize recognition beyond the set of identities used in training. The downsides of this approach are its indirectness and its inefficiency: one has to hope that the bottleneck representation generalizes well to new faces; and by using a bottleneck layer the representation size per face is usually very large (1000s of dimensions). Some recent work [15] has reduced this dimensionality using PCA, but this is a linear transformation that can be easily learnt in one layer of the network.

> 先前基于深度网络的人脸识别方法使用分类层[15,17]，在一组已知的人脸身份上进行训练，然后以中间瓶颈层作为表示，用于概括训练中使用身份以外的识别。这种方法的缺点是其间接性和低效率：人们必须希望瓶颈表示对新面孔具有很好的概括性；通过使用瓶颈层，每个人脸的表示大小通常非常大（1000维）。最近的一些工作[15]使用PCA减少了这种维度，但这是一种线性变换，可以在网络的一层中轻松学习。

In contrast to these approaches, FaceNet directly trains its output to be a compact 128-D embedding using a tripletbased loss function based on LMNN [19]. Our triplets consist of two matching face thumbnails and a non-matching face thumbnail and the loss aims to separate the positive pair from the negative by a distance margin. The thumbnails are tight crops of the face area, no 2D or 3D alignment, other than scale and translation is performed.

> 与这些方法相比，FaceNet直接使用基于LMNN [19]的三元组的损失函数训练其输出，以成为紧凑的128D嵌入式。我们的三元组由两个匹配的人脸缩略图和一个不匹配的人脸缩略图组成，损失函数旨在通过距离边缘将正面对分开。缩略图是人脸区域的紧凑剪切，除了尺度和平移外，不进行2D或3D对齐。

Choosing which triplets to use turns out to be very important for achieving good performance and, inspired by curriculum learning [1], we present a novel online negative exemplar mining strategy which ensures consistently increasing difficulty of triplets as the network trains. To improve clustering accuracy, we also explore hard-positive mining techniques which encourage spherical clusters for the embeddings of a single person.

> 选择哪些三元组使用被证明是获得良好性能的非常重要的因素，受到课程学习 [1] 的启发，我们提出了一种新颖的在线负样本挖掘策略，可以确保随着网络训练的不断进行，三元组的难度不断增加。为了提高聚类的准确性，我们还探索了积极的挖掘技术，以鼓励单个人的嵌入式的球形集群。

As an illustration of the incredible variability that our method can handle see Figure 1. Shown are image pairs from PIE [13] that previously were considered to be very difficult for face verification systems.

> 为了说明我们的方法可以处理的令人难以置信的可变性，请参见图1。显示的是来自 PIE [13] 的图像对，以前被认为对于人脸验证系统非常困难。

An overview of the rest of the paper is as follows: in section 2 we review the literature in this area; section 3.1 defines the triplet loss and section 3.2 describes our novel triplet selection and training procedure; in section 3.3 we describe the model architecture used. Finally in section 4 and 5 we present some quantitative results of our embeddings and also qualitatively explore some clustering results.

> 本文的其余部分概述如下：在第2节中，我们回顾了这一领域的文献；第3.1节定义了三元组损失，第3.2节描述了我们新颖的三元组选择和训练过程；在第3.3节中，我们描述了所使用的模型架构。最后，在第4和5节中，我们提出了一些嵌入的定量结果，并定性地探索了一些聚类结果。

# 2. Related Work
Similarly to other recent works which employ deep networks [15, 17], our approach is a purely data driven method which learns its representation directly from the pixels of the face. Rather than using engineered features, we use a large dataset of labelled faces to attain the appropriate invariances to pose, illumination, and other variational conditions.

> 这段内容说明作者的方法和其他最近使用深度学习的方法[15, 17]类似，是一种纯数据驱动的方法，通过大量标记人脸的数据集学习特征表示。作者的方法不依赖于工程特征，而是直接从人脸的像素学习特征表示，以获得适当的不变性，如姿态、照明等。

In this paper we explore two different deep network architectures that have been recently used to great success in the computer vision community. Both are deep convolutional networks [8, 11]. The first architecture is based on the Zeiler&Fergus [22] model which consists of multiple interleaved layers of convolutions, non-linear activations, local response normalizations, and max pooling layers. We additionally add several 1×1×d convolution layers inspired by the work of [9]. The second architecture is based on the Inception model of Szegedy et al. which was recently used as the winning approach for ImageNet 2014 [16]. These networks use mixed layers that run several different convolutional and pooling layers in parallel and concatenate their responses. We have found that these models can reduce the number of parameters by up to 20 times and have the potential to reduce the number of FLOPS required for comparable performance.

> 本文研究了两种不同的深度网络架构，它们在计算机视觉领域取得了巨大的成功。两者都是深度卷积网络[8, 11]。第一个架构基于Zeiler&Fergus [22]模型，由多层交错的卷积、非线性激活、局部响应归一化和最大池化层组成。我们还受到[9]的启发，添加了几个1×1×d卷积层。第二个架构基于Szegedy等人的Inception模型，最近被用作ImageNet 2014 [16]的获胜方法。这些网络使用混合层，并行运行多个不同的卷积和池化层，并连接它们的响应。我们发现，这些模型可以将参数数量减少多达20倍，并有可能减少所需FLOPS数量以获得相当的性能。

There is a vast corpus of face verification and recognition works. Reviewing it is out of the scope of this paper so we will only briefly discuss the most relevant recent work. The works of [15, 17, 23] all employ a complex system of multiple stages, that combines the output of a deep convolutional network with PCA for dimensionality reduction and an SVM for classification.

> 关于面部验证和识别的巨大语料库存在。因此，本文不会详细评述，我们将简要讨论最相关的最新工作。[15, 17, 23]的工作都采用了多个阶段的复杂系统，将深度卷积网络的输出与PCA的降维以及SVM分类结合起来。

Zhenyao et al. [23] employ a deep network to “warp” faces into a canonical frontal view and then learn CNN that classifies each face as belonging to a known identity. For face verification, PCA on the network output in conjunction with an ensemble of SVMs is used.

> Zhenyao等人 [23]使用深度网络将人脸“变形”为规范正面视图，然后学习CNN，将每个人脸归类为已知身份。对于人脸验证，使用网络输出的PCA与SVM集合。

Taigman et al. [17] propose a multi-stage approach that aligns faces to a general 3D shape model. A multi-class network is trained to perform the face recognition task on over four thousand identities. The authors also experimented with a so called Siamese network where they directly optimize the L1-distance between two face features. Their best performance on LFW (97.35%) stems from an ensemble of three networks using different alignments and color channels. The predicted distances (non-linear SVM predictions based on the $X^2$ kernel) of those networks are combined using a non-linear SVM.

> Taigman等人提出了一种多阶段方法，该方法将面部对齐到一般的3D形状模型。在超过四千个身份上训练多类网络，以执行面部识别任务。作者还试验了一种称为Siamese网络的方法，该方法直接优化两个面部特征的L1距离。他们在LFW上的最佳表现（97.35％）来自三个使用不同对齐和颜色通道的网络的集合。这些网络的预测距离（基于 $X^2$ 内核的非线性SVM预测）通过非线性SVM组合。

Sun et al. [14, 15] propose a compact and therefore relatively cheap to compute network. They use an ensemble of 25 of these network, each operating on a different face patch. For their final performance on LFW (99.47% [15]) the authors combine 50 responses (regular and flipped). Both PCA and a Joint Bayesian model [2] that effectively correspond to a linear transform in the embedding space are employed. Their method does not require explicit 2D/3D alignment. The networks are trained by using a combination of classification and verification loss. The verification loss is similar to the triplet loss we employ [12, 19], in that it minimizes the L2-distance between faces of the same identity and enforces a margin between the distance of faces of different identities. The main difference is that only pairs of images are compared, whereas the triplet loss encourages a relative distance constraint.

> Sun等人提出了一种紧凑且相对便宜的网络。他们使用了25个网络的集合，每个网络在不同的人脸修补上运行。对于LFW的最终表现（99.47% [15]），作者合并了50个响应（正常和翻转）。使用PCA和Joint Bayesian模型[2]，它们实际上对应于嵌入空间中的线性变换。他们的方法不需要明确的2D/3D对齐。通过使用分类和验证损失来训练网络。验证损失与我们使用的三元组损失[12, 19]类似，它最小化相同身份的人脸之间的L2距离，并在不同身份的人脸之间的距离之间强制保持边距。主要的不同之处在于仅比较图像对，而三元组损失鼓励相对距离限制。

A similar loss to the one used here was explored in Wang et al. [18] for ranking images by semantic and visual similarity.

> Wang等人[18]探讨了一个与这里使用的类似的损失，用于通过语义和视觉相似性对图像进行排名。

# 3. Method

FaceNet uses a deep convolutional network. We discuss two different core architectures: The Zeiler&Fergus [22] style networks and the recent Inception [16] type networks. The details of these networks are described in section 3.3.

> FaceNet 使用深度卷积网络。我们讨论了两种不同的核心架构：Zeiler & Fergus [22] 风格的网络和最近的 Inception [16] 类型的网络。这些网络的细节在3.3节中描述。

![在这里插入图片描述](https://img-blog.csdnimg.cn/8e226234ab404b1b96559e0452f64dce.png#pic_center) *Figure 2. Model structure. Our network consists of a batch input layer and a deep CNN followed by L2 normalization, which results in the face embedding. This is followed by the triplet loss during training.*

> 图2: 模型结构。 我们的网络由一个批量输入层和一个深度 CNN 组成，然后是 L2 归一化并用于人脸嵌入。接下来是训练期间的三元组损失。


Given the model details, and treating it as a black box (see Figure 2), the most important part of our approach lies in the end-to-end learning of the whole system. To this end we employ the triplet loss that directly reflects what we want to achieve in face verification, recognition and clustering. Namely, we strive for an embedding $f(x)$, from an image $x$ into a feature space $\mathbb{R}^d$, such that the squared distance between all faces, independent of imaging conditions, of the same identity is small, whereas the squared distance between a pair of face images from different identities is large.

> 鉴于模型的细节，并将其视为黑匣子（见图2），我们方法的最重要部分在于整个系统的端到端学习。为此，我们采用三元组损失，直接反映了我们在人脸验证、识别和聚类中想要实现的目标。即，我们努力寻求一个嵌入 $f(x)$，从图像 $x$ 到特征空间 $\mathbb{R}^d$，使得相同身份的所有人脸，无论成像条件如何，之间的平方距离都很小，而不同身份的人脸图像对之间的平方距离却很大。

![在这里插入图片描述](https://img-blog.csdnimg.cn/554cdbf17b4d4a639b4bb1be6719e440.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
*Figure 3. The Triplet Loss minimizes the distance between an anchor and a positive, both of which have the same identity, and maximizes the distance between the anchor and a negative of a different identity.*

> 图3:  anchor 和 positive 有相同的特征，和 negative 有不同的特征。Triplet Loss 在训练过程中会缩小 anchor 和 postive 的距离，并最大化 anchor 和 negative 的距离。


Although we did not a do direct comparison to other losses, e.g. the one using pairs of positives and negatives, as used in [14] Eq. (2), we believe that the triplet loss is more suitable for face verification. The motivation is that the loss from [14] encourages all faces of one identity to be a projected onto a single point in the embedding space. The triplet loss, however, tries to enforce a margin between each pair of faces from one person to all other faces. This allows the faces for one identity to live on a manifold, while still enforcing the distance and thus discriminability to other identities.

> 尽管我们没有直接与其他损失进行比较，例如 [14] 中使用的正负配对损失，但我们相信三元组损失更适合人脸验证。这的动机在于 [14] 中的损失鼓励将所有身份的脸部投影到特征空间的单个点上。然而，三元组损失试图在一个人的每对脸与所有其他脸之间强制执行间隔。这允许一个身份的脸存在于流形上，同时仍然强制对其他身份的距离和因此可判别性。

The following section describes this triplet loss and how it can be learned efficiently at scale.

> 接下来的章节详细的说明了三元组损失，以及怎么大规模有效地学习它。

## 3.1. Triplet Loss

The embedding is represented by $f(x) \in \mathbb{R}^d$ . It embeds an image $x$ into a d-dimensional Euclidean space. Additionally, we constrain this embedding to live on the d-dimensional hypersphere, i.e. $\left \| f(x) \right \|_2 = 1$. This loss is motivated in [19] in the context of nearest-neighbor classification. Here we want to ensure that an image $x_i^a (anchor)$ of a specific person is closer to all other images $x_i^p (positive)$ of the same person than it is to any image $x_i^n (negative)$ of any other person. This is visualized in Figure 3.

> 嵌入由$f(x)\in\mathbb{R}^d$表示，它将图像$x$嵌入d维欧几里得空间。此外，我们限制此嵌入生存在d维超球面上，即$\left|f(x)\right|_2=1$。该损失在[19]中以最近邻分类的背景提出。在这里，我们希望确保某人的图像$x_i^a(anchor)$比所有其他来自同一人的图像$x_i^p(positive)$更接近，而不是任何其他人的图像$x_i^n(negative)$。这在图3中可视化。

Thus we want:

**Eq. 1**

$$
\left \|  x_i^a - x_i^p  \right \|_2^2 + \alpha < \left \|  x_i^a - x_i^n \right \|_2^2, \ \forall (x_i^a, x_i^p, x_i^n) \in \tau
$$

where $\alpha$ is a margin that is enforced between positive and negative pairs. $\tau$ is the set of all possible triplets in the training set and has cardinality $N$.

> 因此，我们希望
> **Eq. 1**
> $$
> \left \|  x_i^a - x_i^p  \right \|_2^2 + \alpha < \left \|  x_i^a - x_i^n \right \|_2^2, \ \forall (x_i^a, x_i^p, x_i^n) \in \tau
> $$
>其中$\alpha$是在正负对之间强制执行的边界。$\tau$是训练集中所有可能三元组的集合，其基数为$N$。

The loss that is being minimized is then $L =$

**Eq. 2**

$$
\sum_{i}^{N} \left [  \left \|   f(x_{i}^{a}) -  f(x_{i}^{p})  \right \|_2^2  -   \left \|   f(x_{i}^{a}) -  f(x_{i}^{n})  \right \|_2^2  + \alpha \right ]
$$

Generating all possible triplets would result in many triplets that are easily satisfied (i.e. fulfill the constraint in Eq. (1)). These triplets would not contribute to the training and result in slower convergence, as they would still be passed through the network. It is crucial to select hard triplets, that are active and can therefore contribute to improving the model. The following section talks about the different approaches we use for the triplet selection.

> 被最小化的损失是$L=$
> **Eq. 2**
> $$
> \sum_{i}^{N} \left [  \left \|   f(x_{i}^{a}) -  f(x_{i}^{p})  \right \|_2^2  -   \left \|   f(x_{i}^{a}) -  f(x_{i}^{n})  \right \|_2^2  + \alpha \right ]
> $$
> 生成所有可能的三元组将导致许多容易满足的三元组（即满足式(1)中的限制）。这些三元组不会对训练产生贡献，并导致更慢的收敛，因为它们仍将通过网

## 3.2. Triplet Selection

In order to ensure fast convergence it is crucial to select triplets that violate the triplet constraint in Eq. (1). This means that, given $x_i^a$ , we want to select an $x_i^p$ (hard positive) such that $argmax_{x_i^p} \left \| f (x_i^a ) − f (x_i^p ) \right \|_2^2$ and similarly $x_i^n$ (hard negative) such that $argmin_{x_i^n} \left \| f(x_i^a ) − f(x_i^n) \right \|_2^2$. 

> 为了确保快速收敛，选择违反三元组约束（Eq. (1)）的三元组是至关重要的。这意味着，给定$x_i^a$，我们要选择一个$x_i^p$（困难的正样本），使得$argmax_{x_i^p} \left | f (x_i^a ) − f (x_i^p ) \right |2^2$，并且同样的$x_i^n$（困难的负样本），使得$argmin{x_i^n} \left | f(x_i^a ) − f(x_i^n) \right |_2^2$。

It is infeasible to compute the argmin and argmax across the whole training set. Additionally, it might lead to poor training, as mislabelled and poorly imaged faces would dominate the hard positives and negatives. There are two obvious choices that avoid this issue:

* Generate triplets offline every n steps, using the most recent network checkpoint and computing the argmin and argmax on a subset of the data.
* Generate triplets online. This can be done by selecting the hard positive/negative exemplars from within a mini-batch.

Here, we focus on the online generation and use large mini-batches in the order of a few thousand exemplars and only compute the argmin and argmax within a mini-batch.

> 通过整个训练集计算argmin和argmax是不可行的。此外，由于标签错误和图像质量差，它可能导致训练不佳，因为这些因素会占据硬正和负面。有两个明显的选择可以避免这个问题：
> 
> * 离线生成三元组：每隔 n 步生成三元组，使用最近的网络检查点，并在数据的一个子集上计算 argmin 和 argmax。
> * 在线生成三元组：从小批量内选择硬正/负样本生成三元组。
> 
> 在这里，我们专注于在线生成，并在几千个样本中使用大型小批量，只在小批量中计算 argmin 和 argmax。

To have a meaningful representation of the anchor-positive distances, it needs to be ensured that a minimal number of exemplars of any one identity is present in each mini-batch. In our experiments we sample the training data such that around 40 faces are selected per identity per mini-batch. Additionally, randomly sampled negative faces are added to each mini-batch.

> 在每个小批量中，为了保证锚-正确距离具有意义的表示，需要确保每个身份的最小数量的样本在每个小批量中出现。在我们的实验中，我们对训练数据进行抽样，使每个身份每个小批量选择约40张人脸。此外，将随机采样的负面人脸添加到每个小批量中。

Instead of picking the hardest positive, we use all anchor-positive pairs in a mini-batch while still selecting the hard negatives. We don’t have a side-by-side comparison of hard anchor-positive pairs versus all anchor-positive pairs within a mini-batch, but we found in practice that the all anchor-positive method was more stable and converged slightly faster at the beginning of training.

> 我们不再选择最困难的正样本，而是在小批量中使用所有的锚-正样本对，同时仍然选择困难的负样本。我们没有对在小批量内的硬锚-正样本对与所有锚-正样本对进行并列比较，但我们在实践中发现，在训练开始时，所有锚正样本方法更稳定，收敛速度略快。

We also explored the offline generation of triplets in conjunction with the online generation and it may allow the use of smaller batch sizes, but the experiments were inconclusive. Selecting the hardest negatives can in practice lead to bad local minima early on in training, specifically it can result in a collapsed model (i.e. $f(x) = 0$). In order to mitigate this, it helps to select $x_i^n$ such that

**Eq. 3**

$$
\left \|  f(x_i^a) - f(x_i^p) \right \|_2^2 < \left \| f(x_i^a) - f(x_i^n) \right \|_2^2
$$

We call these negative exemplars semi-hard, as they are further away from the anchor than the positive exemplar, but still hard because the squared distance is close to the anchor-positive distance. Those negatives lie inside the margin $\alpha$.

> 我们还探索了离线生成三元组和在线生成三元组的结合，它可能允许使用较小的批次大小，但实验结果不确定。选择最困难的负面样本在实践中可能导致训练初期的坏局部极值，特别是可能导致模型崩溃（即 $f(x) = 0$）。为了缓解这种情况，选择 $x_i^n$ 使得
> 
> **Eq. 3**
> $$
> \left \|  f(x_i^a) - f(x_i^p) \right \|_2^2 < \left \| f(x_i^a) - f(x_i^n) \right \|_2^2
> $$
> 这些负面样本比锚点远，但仍然是困难的，因为平方距离接近锚点-正面样本距离。这些负面样本在马赛克 $\alpha$ 内。我们称这些负面样本为半困难的样本，因为它们离锚点更远，但仍然是困难的，因为平方距离接近锚点-正样本距离。

As mentioned before, correct triplet selection is crucial for fast convergence. On the one hand we would like to use small mini-batches as these tend to improve convergence during Stochastic Gradient Descent (SGD) [20]. On the other hand, implementation details make batches of tens to hundreds of exemplars more efficient. The main constraint with regards to the batch size, however, is the way we select hard relevant triplets from within the mini-batches. In most experiments we use a batch size of around 1,800 exemplars.

> 选择正确的三元组对于快速收敛至关重要。一方面，我们希望使用小的小批量数据，因为这些通常会在随机梯度下降（SGD）期间改善收敛[20]。另一方面，实现细节使得几十到几百个样本的批次效率更高。然而，与批量大小相关的主要约束是我们如何从小批量数据中选择硬相关三元组。在大多数实验中，我们使用大约1800个样本的批量大小。

## 3.3. Deep Convolutional Networks
In all our experiments we train the CNN using Stochastic Gradient Descent (SGD) with standard backprop [8, 11] and AdaGrad [5]. In most experiments we start with a learning rate of 0.05 which we lower to finalize the model. The models are initialized from random, similar to [16], and trained on a CPU cluster for 1,000 to 2,000 hours. The decrease in the loss (and increase in accuracy) slows down drastically after 500h of training, but additional training can still significantly improve performance. The margin $\alpha$ is set to 0.2.

> 在所有的实验中，我们使用随机梯度下降(SGD)，搭配标准反向传播[8,11]和AdaGrad [5]来训练卷积神经网络。在大多数实验中，我们从学习率为0.05开始，最后降低学习率以完成模型。模型初始化为随机，类似于[16]，并在CPU集群上训练1,000到2,000个小时。在训练500小时后，损失的降低（以及准确率的提高）会急剧减缓，但是额外的训练仍然可以显着提高性能。间隔α设置为0.2。

We used two types of architectures and explore their trade-offs in more detail in the experimental section. Their practical differences lie in the difference of parameters and FLOPS. The best model may be different depending on the application. E.g. a model running in a datacenter can have many parameters and require a large number of FLOPS, whereas a model running on a mobile phone needs to have few parameters, so that it can fit into memory. All our models use rectified linear units as the non-linear activation function.

> 我们使用了两种类型的架构，并在实验部分详细探讨了它们的权衡。它们的实际差别在于参数和FLOPS的差异。最佳模型可能因应用程序而异。例如，运行在数据中心的模型可以具有许多参数，需要大量的FLOPS，而运行在移动电话上的模型需要具有很少的参数，以便能够适应内存。我们所有的模型都使用整流线性单元作为非线性激活函数。
![在这里插入图片描述](https://img-blog.csdnimg.cn/6aa3df0c5c4f4de68394a05cc9885625.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)*Table 1. NN1. This table show the structure of our Zeiler&Fergus [22] based model with $1 \times 1$ convolutions in- spired by [9]. The input and output sizes are described in $rows \times cols \times filters$. The kernel is specified as $rows \times cols$, stride and the maxout [6] pooling size as $p = 2$.*

> 表 1. NN1。 该表显示了我们基于 Zeiler&Fergus [22] 的模型的结构，该模型具有受 [9] 启发的 $1 \times 1$ 卷积。 输入和输出大小以 $rows \times cols \times filters$ 描述。 内核被指定为 $rows \times cols$、stride 和 maxout [6] 池大小为 $p = 2$。

The first category, shown in Table 1, adds $1 \times 1 \times d$ convolutional layers, as suggested in [9], between the standard convolutional layers of the Zeiler&Fergus [22] architecture and results in a model 22 layers deep. It has a total of 140 million parameters and requires around 1.6 billion FLOPS per image.

> 第一类，如表1所示，在Zeiler＆Fergus [22]架构的标准卷积层之间，根据[9]的建议，添加$1 \times 1 \times d$卷积层，导致模型深22层。它共有1.4亿个参数，每张图像需要约16亿FLOPS。

The second category we use is based on GoogLeNet style Inception models [16]. These models have $20 \times$ fewer parameters (around 6.6M-7.5M) and up to 5× fewer FLOPS (between 500M-1.6B). Some of these models are dramatically reduced in size (both depth and number of filters), so that they can be run on a mobile phone. One, NNS1, has 26M parameters and only requires 220M FLOPS per image. The other, NNS2, has 4.3M parameters and 20M FLOPS. Table 2 describes NN2 our largest network in detail. NN3 is identical in architecture but has a reduced input size of $160 \times 160$. NN4 has an input size of only $96 \times 96$, thereby drastically reducing the CPU requirements (285M FLOPS vs 1.6B for NN2). In addition to the reduced input size it does not use $5 \times 5$ convolutions in the higher layers as the receptive field is already too small by then. Generally we found that the $5 \times 5$ convolutions can be removed throughout with only a minor drop in accuracy. Figure 4 compares all our models.

> 我们使用的第二类基于GoogLeNet风格的Inception模型（[16]）。这些模型的参数要少$20$倍（约$6.6M-7.5M$），FLOPS也要少$5$倍（在$500M-1.6B$之间）。其中一些模型的大小已经明显减小（深度和滤波器数量），以便在手机上运行。一个模型（NNS1）有26M个参数，每张图像仅需220M FLOPS。另一个（NNS2）有4.3M个参数和20M FLOPS。表2详细描述了我们的最大网络NN2。NN3的体系结构与NN2相同，但输入大小为$160 \times 160$。NN4的输入大小仅为$96 \times 96$，从而大大减小了CPU的要求（285M FLOPS vs 1.6B for NN2）。除了输入大小的减小外，在较高层没有使用$5 \times 5$的卷积，因为感受野已经太小了。通常，我们发现可以在所有模型中删除$5 \times 5$的卷积，仅有一个小的精度下降。图4比较了所有模型。



![在这里插入图片描述](https://img-blog.csdnimg.cn/d0d9da264fd94c059e9ee9de5601f38a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
*Table 2. NN2. Details of the NN2 Inception incarnation. This model is almost identical to the one described in [16]. The two major differences are the use of L2 pooling instead of max pooling (m), where specified. The pooling is always $3 \times 3$ (aside from the final average pooling) and in parallel to the convolutional modules inside each Inception module. If there is a dimensionality reduction after the pooling it is denoted with $p$. $1 \times 1$, $3 \times 3$, and $5 \times 5$ pooling are then concatenated to get the final output.*

> 表2. NN2. NN2 Inception的详细信息。该模型和[16]中描述的很相似。最主要的两个区别是在L2用池化方法替代了最大池化方法。池化总是$3 \times 3$大小（除了最终的平均池化）并与每个 Inception 模块内的卷积模块并行。 如果在池化之后有降维，则用 $p$ 表示。 然后将 $1\times 1$、$3\times 3$ 和 $5\times 5$ 池连接起来以获得最终输出。


# 4. Datasets and Evaluation

We evaluate our method on four datasets and with the exception of Labelled Faces in the Wild and YouTube Faces we evaluate our method on the face verification task. I.e. given a pair of two face images a squared L2 distance threshold $D(x_i, x_j)$ is used to determine the classification of same and different. All faces pairs $(i, j)$ of the same identity are denoted with $P_{same}$, whereas all pairs of different identities are denoted with $P_{diff}$.

> 我们在四个数据集上评估了我们的方法，除了LFW和YouTube Faces以外，我们都在人脸验证任务上评估了我们的方法。即，给定两个人脸图像对，使用平方L2距离阈值$D(x_i, x_j)$来确定相同和不同的分类。相同身份的所有人脸对$(i,j)$被表示为$P_{same}$，而不同身份的所有对被表示为$P_{diff}$。

We define the set of all *true accepts* as

**Eq. 4**

$$
TA(d) = \{  (i, j) \in P_{same}, \ with \ D(x_i, x_j) \leq d \}
$$

These are the face pairs $(i, j)$ that were correctly classified as same at threshold d. Similarly

> 我们定义所有“真实接受”的集合为：
> **Eq. 4**
> $$
> TA(d) = \{  (i, j) \in P_{same}, \ with \ D(x_i, x_j) \leq d \}
> $$
> 这些是在阈值d处被正确分类为相同的人脸对$(i, j)$。同样，我们定义所有“假阳性”为：

**Eq. 5**

$$
FA(d) = \{  (i, j) \in P_{diff}, \ with \ D(x_i, x_j) \leq d \}
$$

is the set of all pairs that was incorrectly classified as same (false accept).

> **Eq. 5**
> $$
> FA(d) = \{  (i, j) \in P_{diff}, \ with \ D(x_i, x_j) \leq d \}
> $$
> 是所有错误地被分类为相同的对（错误接受）的集合。

The validation rate VAL(d) and the false accept rate FAR(d) for a given face distance d are then defined as

**Eq. 6**

$$
VAL(d) = \frac{| TA(d) |}{| P_{same} |}, \ FAR(d) = \frac{| FA(d) |}{| P_{diff} |}
$$

> 这样，给定一个距离阈值 d 的验证率 VAL(d) 和误识率 FAR(d) 就定义为：
> **Eq. 6**
> $$
> VAL(d) = \frac{| TA(d) |}{| P_{same} |}, \ FAR(d) = \frac{| FA(d) |}{| P_{diff} |}
> $$

## 4.1. Hold-out Test Set

We keep a hold out set of around one million images, that has the same distribution as our training set, but dis- joint identities. For evaluation we split it into five disjoint sets of 200k images each. The FAR and VAL rate are then computed on $100k \times 100k$ image pairs. Standard error is reported across the five splits.

> 我们保留了一个包含大约一百万张图像的测试集，该测试集具有与训练集相同的分布，但身份不同。为了评估，我们将其分为五个不相交的200k图像集。然后在$100k \times 100k$图像对上计算FAR和VAL率。标准误差在五个分裂中报告。

## 4.2. Personal Photos

Thisisatestsetwithsimilardistributiontoourtraining set, but has been manually verified to have very clean labels. It consists of three personal photo collections with a total of around 12k images. We compute the FAR and VAL rate across all 12k squared pairs of images.

> 这是一个与训练集相似的测试集，但已经经过人工验证具有非常干净的标签。它包含三个个人照片集，共计约12k张图像。我们计算所有12k平方图像对的FAR和VAL率。

## 4.3. Academic Datasets

Labeled Faces in the Wild (LFW) is the de-facto aca- demic test set for face verification [7]. We follow the stan- dard protocol for unrestricted, labeled outside data and re- port the mean classification accuracy as well as the standard error of the mean.

> Labeled Faces in the Wild（LFW）是人脸验证的默认学术测试集[7]。我们遵循未受限制的标记外部数据的标准协议，并报告平均分类准确率以及平均标准误差。

Youtube Faces DB [21] is a new dataset that has gained popularity in the face recognition community [17, 15]. The setup is similar to LFW, but instead of verifying pairs of images, pairs of videos are used.

> YouTube Faces DB [21] 是一个在人脸识别社区中越来越受欢迎的新数据集 [17,15]。该数据集的设置与 LFW 相似，但使用的是视频对，而不是图像对进行验证。

# 5. Experiments

If not mentioned otherwise we use between 100M-200M training face thumbnails consisting of about 8M different identities. A face detector is run on each image and a tight bounding box around each face is generated. These face thumbnails are resized to the input size of the respective network. Input sizes range from $96 \times 96$ pixels to $224 \times 224$ pixels in our experiments.

> 在本文中，除非另有说明，我们使用了100M至200M个训练人脸缩略图，其中包含约8M个不同的身份。我们在每个图像上运行人脸检测器，并生成每个人脸周围的紧密边界框。将这些人脸缩略图调整为相应网络的输入大小。我们实验中的输入大小从96 x 96像素到224 x 224像素不等。

## 5.1. Computation Accuracy Trade-off

Before diving into the details of more specific experiments lets discuss the trade-off of accuracy versus number of FLOPS that a particular model requires. Figure 4 shows the FLOPS on the x-axis and the accuracy at 0.001 false accept rate (FAR) on our user labelled test-data set from section 4.2. It is interesting to see the strong correlation between the computation a model requires and the accuracy it achieves. The figure highlights the five models (NN1, NN2, NN3, NNS1, NNS2) that we discuss in more detail in our experiments.

> 在深入研究具体实验之前，让我们讨论一下特定模型所需精度和 FLOPS 之间的平衡。图 4 显示了 X 轴上的 FLOPS 和在我们的用户标记测试数据集（见 4.2 节）上以 0.001 false accept rate（FAR）为标准的准确性。有趣的是，模型所需的计算量与它的准确性存在很强的相关性。该图突出显示了我们在实验中详细讨论的五种模型（NN1，NN2，NN3，NNS1，NNS2）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/dce8ad9cc9d24aae9e5d578937e71ea9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
*Figure 4. FLOPS vs. Accuracy trade-off. Shown is the trade-off between FLOPS and accuracy for a wide range of different model sizes and architectures. Highlighted are the four models that we focus on in our experiments.*

> 图 4. FLOPS 与准确性的权衡。 显示的是各种不同模型大小和架构的 FLOPS 和准确性之间的权衡。 突出显示的是我们在实验中关注的四个模型。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2f59289e748b47c882a9a7d86c14b9b5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
*Table 3. Network Architectures. This table compares the per- formance of our model architectures on the hold out test set (see section 4.1). Reported is the mean validation rate VAL at 10E-3 false accept rate. Also shown is the standard error of the mean across the five test splits.*

> 表 3. 网络架构。 该表比较了我们的模型架构在保持测试集上的性能（参见第 4.1 节）。 报告的是 10E-3 错误接受率的平均验证率 VAL。 还显示了五个测试拆分的平均值的标准误差。

We also looked into the accuracy trade-off with regards to the number of model parameters. However, the picture is not as clear in that case. For example, the Inception based model NN2 achieves a comparable performance to NN1, but only has a 20th of the parameters. The number of FLOPS is comparable, though. Obviously at some point the performance is expected to decrease, if the number of parameters is reduced further. Other model architectures may allow further reductions without loss of accuracy, just like Inception [16] did in this case.

> 此外，我们还研究了模型参数数量与准确性之间的权衡。然而，情况并不那么明朗。例如，基于Inception的模型NN2在数量上与NN1相当，但只有NN1的1/20。FLOPS数量也相当。显然，如果进一步减少参数的数量，性能将有所下降。其他模型架构可能允许在不损失准确性的情况下进一步减少，就像Inception [16]在这种情况下一样。

![在这里插入图片描述](https://img-blog.csdnimg.cn/898dbf78d97644e0aa23cc07045585fe.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
*Figure 5. Network Architectures. This plot shows the com- plete ROC for the four different models on our personal pho- tos test set from section 4.2. The sharp drop at 10E-4 FAR can be explained by noise in the groundtruth labels. The mod- els in order of performance are: NN2: $224 \times 224$ input Inception based model; NN1: Zeiler&Fergus based network with $1 \times 1$ convolutions; NNS1: small Inception style model with only 220M FLOPS; NNS2: tiny Inception model with only 20M FLOPS.*

> 图 5. 网络架构。 该图显示了第 4.2 节中我们的个人照片测试集上四种不同模型的完整 ROC。 10E-4 FAR 的急剧下降可以用 groundtruth 标签中的噪声来解释。 模型按性能排序为： NN2：$224 \times 224$ 输入基于 Inception 的模型； NN1：基于 Zeiler&Fergus 的 $1 \times 1$ 卷积网络； NNS1：只有 220M FLOPS 的小型 Inception 风格模型； NNS2：只有 20M FLOPS 的小型 Inception 模型。


## 5.2. Effect of CNN Model

We now discuss the performance of our four selected models in more detail. On the one hand we have our tradi- tional Zeiler&Fergus based architecture with $1 \times 1$ convolutions [22, 9] (see Table 1). On the other hand we have Inception [16] based models that dramatically reduce the model size. Overall, in the final performance the top models of both architectures perform comparably. However, some of our Inception based models, such as NN3, still achieve good performance while significantly reducing both the FLOPS and the model size.

> 现在，我们将详细讨论我们选择的四个模型的性能。一方面，我们有基于Zeiler&Fergus的传统架构，带有$1 \times 1$卷积22,9。另一方面，我们有基于Inception[16]的模型，可以大大减小模型的大小。总的来说，最终的性能两种架构的顶级模型的表现可以相媲美。然而，我们的一些基于Inception的模型，如NN3，仍然能够在显著减小FLOPS和模型大小的同时实现良好的性能。

The detailed evaluation on our personal photos test set is shown in Figure 5. While the largest model achieves a dramatic improvement in accuracy compared to the tiny NNS2, the latter can be run 30ms / image on a mobile phone and is still accurate enough to be used in face clustering. The sharp drop in the ROC for FAR < 10−4 indicates noisy labels in the test data groundtruth. At extremely low false accept rates a single mislabeled image can have a significant impact on the curve.

> 细致的评估结果在个人照片测试集上显示如图5。尽管最大模型相对于小模型NNS2在准确性上有了戏剧性的提高，但后者在移动手机上的每张图片运行时间仅为30毫秒，仍然足够准确，可用于人脸聚类。ROC在FAR < 10^-4时的急剧下降表明测试数据地面实况中存在噪声标签。在极低的错误接受率下，一张标签错误的图像可能会对曲线产生重大影响。


## 5.3. Sensitivity to Image Quality

Table 4 shows the robustness of our model across a wide range of image sizes. The network is surprisingly robust with respect to JPEG compression and performs very well down to a JPEG quality of 20. The performance drop is very small for face thumbnails down to a size of $120 \times 120$ pixels and even at $80 \times 80$ pixels it shows acceptable performance. This is notable, because the network was trained on $220 \times 220$ input images. Training with lower resolution faces could improve this range further.

> 表4显示了我们的模型在各种大小的图像上的稳健性。该网络对JPEG压缩非常稳健，在JPEG质量为20时仍表现出色。即使对于$120 \times 120$像素的人脸缩略图，性能下降也非常小，甚至在$80 \times 80$像素时仍表现出可接受的性能。这是值得注意的，因为该网络是在$220 \times 220$输入图像上训练的。使用更低分辨率的人脸训练可以进一步提高这个范围。


![在这里插入图片描述](https://img-blog.csdnimg.cn/144991862e964160accef037568847cb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)*Table 4. Image Quality. The table on the left shows the effect on the validation rate at 10E-3 precision with varying JPEG quality. The one on the right shows how the image size in pixels effects the validation rate at 10E-3 precision. This experiment was done with NN1 on the first split of our test hold-out dataset.*

> 表 4. 图像质量。 左侧的表格显示了 10E-3 精度下不同 JPEG 质量对验证率的影响。 右边的一个显示了以像素为单位的图像大小如何影响 10E-3 精度的验证率。 该实验是在我们的测试保留数据集的第一次拆分上使用 NN1 完成的。


![在这里插入图片描述](https://img-blog.csdnimg.cn/ea52e122984d47b4b0aa69c8bfd328d4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_13,color_FFFFFF,t_70,g_se,x_16#pic_center)*Table 5. Embedding Dimensionality. This Table compares the effect of the embedding dimensionality of our model NN1 on our hold-out set from section 4.1. In addition to the VAL at 10E-3 we also show the standard error of the mean computed across five splits.*

> 表 5. 嵌入维度。 该表比较了我们的模型 NN1 的嵌入维度对第 4.1 节中的保留集的影响。 除了 10E-3 处的 VAL，我们还显示了跨五个拆分计算的平均值的标准误差。

## 5.4.  Embedding Dimensionality
We explored various embedding dimensionalities and se- lected 128 for all experiments other than the comparison re- ported in Table 5. One would expect the larger embeddings to perform at least as good as the smaller ones, however, it is possible that they require more training to achieve the same accuracy. That said, the differences in the performance re- ported in Table 5 are statistically insignificant.

> 我们探索了各种嵌入维度，并在除表5中报告的比较外，为所有实验选择了128维。人们期望较大的嵌入效果至少与较小的嵌入效果相同，但它们可能需要更多的训练才能达到相同的准确性。尽管如此，表5中报告的性能差异在统计上是不显著的。

It should be noted, that during training a 128 dimensional float vector is used, but it can be quantized to 128-bytes without loss of accuracy. Thus each face is compactly represented by a 128 dimensional byte vector, which is ideal for large scale clustering and recognition. Smaller embed- dings are possible at a minor loss of accuracy and could be employed on mobile devices.

> 应该注意的是，在训练期间使用的是128维的浮点向量，但它可以量化为128字节而不损失精度。因此，每个人脸都由128维字节向量紧凑地表示，这对大规模聚类和识别非常理想。较小的嵌入可以在较小的精度损失下实现，并可在移动设备上使用。

## 5.5. Amount of Training Data

Table 6 shows the impact of large amounts of training data. Due to time constraints this evaluation was run on a smaller model; the effect may be even larger on larger models. It is clear that using tens of millions of exemplars results in a clear boost of accuracy on our personal photo test set from section 4.2. Compared to only millions of images the relative reduction in error is 60%. Using another order of magnitude more images (hundreds of millions) still gives a small boost, but the improvement tapers off.

> 表6显示了大量训练数据的影响。由于时间限制，这个评估在较小的模型上进行；在更大的模型上的效果可能更大。很明显，使用数百万张样本在我们第4.2节的个人照片测试集上显示出明显的精度提高。与仅数百万张图像相比，误差的相对减少是60%。再使用另一个数量级的图像（数以亿计的图像）仍然有小幅提高，但改进逐渐减缓。


![在这里插入图片描述](https://img-blog.csdnimg.cn/77d3a66132bc437594e2eab176449fc0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_16,color_FFFFFF,t_70,g_se,x_16#pic_center)*Table 6. Training Data Size. This table compares the performance after 700h of training for a smaller model with $96 \times 96$ pixel inputs. The model architecture is similar to NN2, but without the $5 \times 5$ con- volutions in the Inception modules.*

> 表 6. 训练数据大小。 此表比较了具有 $96 \times 96$ 像素输入的较小模型在训练 700 小时后的性能。 模型架构类似于 NN2，但在 Inception 模块中没有 $5 \times 5$ 卷积。

![在这里插入图片描述](https://img-blog.csdnimg.cn/56348f1775244be88e13e8bd4577ccdd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)*Figure 6. LFW errors. This shows all pairs of images that were incorrectly classified on LFW.*

> 图 6. LFW 错误。 这显示了在 LFW 上被错误分类的所有图像对。


## 5.6. Performance on LFW

We evaluate our model on LFW using the standard pro- tocol for unrestricted, labeled outside data. Nine training splits are used to select the L2-distance threshold. Classification (same or different) is then performed on the tenth test split. The selected optimal threshold is 1.242 for all test splits except split eighth (1.256). Our model is evaluated in two modes:

1. Fixed center crop of the LFW provided thumbnail.
2. Aproprietaryfacedetector(similartoPicasa[3])isrun on the provided LFW thumbnails. If it fails to align the face (this happens for two images), the LFW alignment is used.

> 我们使用标准协议评估模型在 LFW 上的表现，这是一个不受限制，外部标记数据。我们使用九次训练分割来选择 L2 距离阈值。然后在第十次测试分割上进行分类（相同或不同）。除第八次分割外，所有测试分割的选定最佳阈值均为 1.242。我们的模型在两种模式下评估：
> 1. 固定的 LFW 缩略图中心剪切。
> 2. 运行专有人脸检测器（类似于 Picasa [3]）的 LFW 缩略图。如果无法对齐人脸（这发生在两张图像上），则使用 LFW 对齐。

Figure 6 gives an overview of all failure cases. It shows false accepts on the top as well as false rejects at the bottom. We achieve a classification accuracy of $98.87 \% \pm 0.15$ when using the fixed center crop described in (1) and the record breaking $99.63 \% \pm 0.09$ standard error of the mean when using the extra face alignment (2). This reduces the error reported for DeepFace in [17] by more than a factor of 7 and the previous state-of-the-art reported for DeepId2+ in [15] by $30 \%$. This is the performance of model NN1, but even the much smaller NN3 achieves performance that is not statistically significantly different.

> 图 6 给出了所有失败情况的概述。它显示了顶部的错误接受和底部的错误拒绝。当使用 (1) 中描述的固定中心裁剪时，我们获得 $98.87 % \pm 0.15$ 的分类准确性，并使用额外的面部对齐 (2) 时获得了创纪录的 $99.63 % \pm 0.09$ 标准误差的平均值。这将 [17] 中报告的 DeepFace 的错误减少了 7 倍，并将 [15] 报告的 DeepId2+ 的错误减少了 $30 %$。这是 NN1 模型的性能，但即使是更小的 NN3 模型也可以获得不统计显著差异的性能。

## 5.7. Performance on Youtube Faces DB

We use the average similarity of all pairs of the first one hundred frames that our face detector detects in each video. This gives us a classification accuracy of $95.12 \% \pm 0.39$. Using the first one thousand frames results in $95.18 \%$. Compared to [17] $91.4 \%$ who also evaluate one hundred frames per video we reduce the error rate by almost half. DeepId2+ [15] achieved $93.2 \%$ and our method reduces this error by $30 \%$, comparable to our improvement on LFW.

> 我们使用我们的人脸检测器在每个视频中检测的前100帧的所有对的平均相似度。这给了我们$95.12 % \pm 0.39$的分类准确率。使用前1000帧的结果为$95.18 %$。与[17]评估每个视频100帧的$91.4 %$相比，我们减少了近一半的错误率。DeepId2+ [15] 实现了 $93.2 %$，我们的方法将其错误率降低了 $30 %$，与我们在LFW上的改进相当。


## 5.8. Face Clustering

Our compact embedding lends itself to be used in order to cluster a users personal photos into groups of people with the same identity. The constraints in assignment imposed by clustering faces, compared to the pure verification task, lead to truly amazing results. Figure 7 shows one cluster in a users personal photo collection, generated using agglom- erative clustering. It is a clear showcase of the incredible invariance to occlusion, lighting, pose and even age.

> 我们紧凑的嵌入很适合用于将用户的个人照片分组为具有相同身份的人群。与纯验证任务相比，聚类面部带来的任务限制导致了令人惊奇的结果。图7显示了用户个人照片集合中的一个聚类，是使用凝聚聚类生成的。这是对遮挡、照明、姿势甚至年龄的惊人不变性的清晰展示。

# 6. Summary

We provide a method to directly learn an embedding into an Euclidean space for face verification. This sets it apart from other methods [15, 17] who use the CNN bottleneck layer, or require additional post-processing such as concate- nation of multiple models and PCA, as well as SVM clas- sification. Our end-to-end training both simplifies the setup and shows that directly optimizing a loss relevant to the task at hand improves performance.

> 我们提供了一种直接学习人脸验证的欧几里得空间嵌入的方法。这与其他方法[15, 17]不同，它们使用CNN瓶颈层，或需要额外的后处理，如多个模型的连接和PCA以及SVM分类。我们的端到端培训简化了设置并表明直接优化与任务相关的损失可以提高性能。 

Another strength of our model is that it only requires minimal alignment (tight crop around the face area). [17], for example, performs a complex 3D alignment. We also experimented with a similarity transform alignment and no- tice that this can actually improve performance slightly. It is not clear if it is worth the extra complexity.

> 我们的模型的另一个优势是只需要最小的对齐（对脸部区域进行紧密的裁剪）。例如，[17]执行复杂的3D对齐。我们还尝试了相似变换对齐，并发现这实际上可以略微提高性能。不清楚是否值得额外的复杂性。

Future work will focus on better understanding of the error cases, further improving the model, and also reducing model size and reducing CPU requirements. We will also look into ways of improving the currently extremely long training times, e.g. variations of our curriculum learn- ing with smaller batch sizes and offline as well as online positive and negative mining.

> 未来的工作将着重于更好地了解错误情况，进一步改进模型，同时减小模型大小并减小CPU要求。我们还将研究如何改进当前极长的培训时间，例如使用更小的批量大小和离线以及在线正负挖掘的课程学习的变化。

![在这里插入图片描述](https://img-blog.csdnimg.cn/18906728e3b64cbab5132b127107a265.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_13,color_FFFFFF,t_70,g_se,x_16#pic_center)
*Figure 7. Face Clustering. Shown is an exemplar cluster for one user. All these images in the users personal photo collection were clustered together.*

> 图 7. 人脸聚类。 显示的是一个用户的示例集群。 用户个人照片集中的所有这些图像都聚集在一起。

# Acknowledgments

We would like to thank Johannes Steffens for his discus- sions and great insights on face recognition and Christian Szegedy for providing new network architectures like [16] and discussing network design choices. Also we are in- debted to the DistBelief [4] team for their support espe- cially to Rajat Monga for help in setting up efficient training schemes.

Also our work would not have been possible without the support of Chuck Rosenberg, Hartwig Adam, and Simon Han.


# References
[1] Y. Bengio, J. Louradour, R. Collobert, and J. Weston. Cur- riculum learning. In Proc. of ICML, New York, NY, USA, 2009. 2
[2] D. Chen, X. Cao, L. Wang, F. Wen, and J. Sun. Bayesian face revisited: A joint formulation. In Proc. ECCV, 2012. 2
[3] D. Chen, S. Ren, Y. Wei, X. Cao, and J. Sun. Joint cascade face detection and alignment. In Proc. ECCV, 2014. 8
[4] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, M. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, Q. V. Le, and A. Y. Ng. Large scale distributed deep networks. In P. Bartlett, F. Pereira, C. Burges, L. Bottou, and K. Wein-berger, editors, NIPS, pages 1232–1240. 2012. 9
[5] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. J. Mach. Learn. Res., 12:2121–2159, July 2011. 4
[6] I. J. Goodfellow, D. Warde-farley, M. Mirza, A. Courville, and Y. Bengio. Maxout networks. In In ICML, 2013. 4
[7] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Technical Re- port 07-49, University of Massachusetts, Amherst, October 2007. 5
[8] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Compu- tation, 1(4):541–551, Dec. 1989. 2, 4
[9] M. Lin, Q. Chen, and S. Yan. Network in network. CoRR, abs/1312.4400, 2013. 2, 4, 6
[10] C. Lu and X. Tang. Surpassing human-level face veri- fication performance on LFW with gaussianface. CoRR, abs/1404.3840, 2014. 1
[11] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature, 1986. 2, 4
[12] M.SchultzandT.Joachims.Learningadistancemetricfrom relative comparisons. In S. Thrun, L. Saul, and B. Schölkopf, editors, NIPS, pages 41–48. MIT Press, 2004. 2
[13] T.Sim,S.Baker,andM.Bsat.TheCMUpose,illumination, and expression (PIE) database. In In Proc. FG, 2002. 2
[14] Y. Sun, X. Wang, and X. Tang. Deep learning face
representation by joint identification-verification. CoRR, abs/1406.4773, 2014. 1, 2, 3
[15] Y. Sun, X. Wang, and X. Tang. Deeply learned face representations are sparse, selective, and robust. CoRR, abs/1412.1265, 2014. 1, 2, 5, 8
[16] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. CoRR, abs/1409.4842, 2014.2,4,5,6,9
[17] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf. Deepface: Closing the gap to human-level performance in face verifica- tion. In IEEE Conf. on CVPR, 2014. 1, 2, 5, 8
[18] J. Wang, Y. Song, T. Leung, C. Rosenberg, J. Wang, J. Philbin, B. Chen, and Y. Wu. Learning fine-grained image similarity with deep ranking. CoRR, abs/1404.4661, 2014. 2
[19] K.Q.Weinberger,J.Blitzer,andL.K.Saul.Distancemetric learning for large margin nearest neighbor classification. In NIPS. MIT Press, 2006. 2, 3
[20] D. R. Wilson and T. R. Martinez. The general inefficiency of batch training for gradient descent learning. Neural Net- works, 16(10):1429–1451, 2003. 4
[21] L. Wolf, T. Hassner, and I. Maoz. Face recognition in un- constrained videos with matched background similarity. In IEEE Conf. on CVPR, 2011. 5
[22] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. CoRR, abs/1311.2901, 2013. 2, 4, 6 [23] Z. Zhu, P. Luo, X. Wang, and X. Tang. Recover canonical- view faces in the wild with deep neural networks. CoRR, abs/1404.3543, 2014. 2