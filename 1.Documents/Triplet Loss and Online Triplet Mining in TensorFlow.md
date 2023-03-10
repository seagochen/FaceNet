# Authors and Publishment
## Authors
Olivier Moindrot

## Original
https://omoindrot.github.io/triplet-loss

# Text

In face recognition, triplet loss is used to learn good embeddings (or “encodings”) of faces. If you are not familiar with triplet loss, you should first learn about it by watching this coursera video from Andrew Ng’s deep learning specialization.

> 在人脸识别中，triplet loss 用于学习人脸“编码”。 如果您不熟悉三元组损失，您应该首先通过观看 Andrew Ng 的深度学习专业的课程视频来了解它。

Triplet loss is known to be difficult to implement, especially if you add the constraints of building a computational graph in TensorFlow.

> 众所周知，Triplet loss 很难实现，尤其是当您在 TensorFlow 中添加构建计算图的约束时。

In this post, I will define the triplet loss and the different strategies to sample triplets. I will then explain how to correctly implement triplet loss with online triplet mining in TensorFlow.

> 在这篇文章中，我将定义三元组损失和不同的采样策略。 然后我将解释如何在 TensorFlow 中正确实现三元组损失。

About two years ago, I was working on face recognition during my internship at Reminiz and I answered a question on stackoverflow about implementing triplet loss in TensorFlow. I concluded by saying:

> 大约两年前，我在 Reminiz 从事人脸识别方面的实习期间，在 stackoverflow 上回答了一个关于在 TensorFlow 中实现三元组损失的问题。 我总结说：

*Clearly, implementing triplet loss in Tensorflow is hard, and there are ways to make it more efficient than sampling in python but explaining them would require a whole blog post !*

> 显然，在 Tensorflow 中实现三元组损失是很困难的，并且有一些方法可以使其比在 python 中的采样更有效，但解释它们需要一篇完整的博客文章！

Two years later, here we go.

> 两年后，来了！

All the code can be found on this [github repository](https://github.com/omoindrot/tensorflow-triplet-loss).

> 并且你可以在github连接中找到实现代码。

## Triplet loss and triplet mining
### Why not just use softmax?
> 为什么不只使用softmax

The triplet loss for face recognition has been introduced by the paper FaceNet: A Unified Embedding for Face Recognition and Clustering from Google. They describe a new approach to train face embeddings using online triplet mining, which will be discussed in the next section.

> 用于人脸识别的三元组损失已由 Google 的论文 FaceNet: A Unified Embedding for Face Recognition and Clustering 引入。 他们描述了一种使用在线三元组挖掘训练人脸嵌入的新方法，这将在下一节中讨论。

Usually in supervised learning we have a fixed number of classes and train the network using the softmax cross entropy loss. However in some cases we need to be able to have a variable number of classes. In face recognition for instance, we need to be able to compare two unknown faces and say whether they are from the same person or not.

> 通常在监督学习中，我们有固定数量的类，并使用 softmax 交叉熵损失来训练网络。 但是在某些情况下，我们需要能够拥有可变数量的类。 例如，在人脸识别中，我们需要能够比较两张未知的人脸，并判断它们是否来自同一个人。

Triplet loss in this case is a way to learn good embeddings for each face. In the embedding space, faces from the same person should be close together and form well separated clusters.


## Definition of the loss

>![在这里插入图片描述](./imgs/triplet_loss.png)
> Triplet loss on two positive faces (Obama) and one negative face (Macron)
> **As the reason for image censorship, I covered the faces with colour. If you want to check the original picture, go and open up the link of the original post here https://omoindrot.github.io/triplet-loss** 
>

The goal of the triplet loss is to make sure that:
> 三元组损失的目标是确保：

Two examples with the same label have their embeddings close together in the embedding space
Two examples with different labels have their embeddings far away.

> 具有相同标签的两个示例在嵌入空间中的嵌入很接近 具有不同标签的两个示例的嵌入很远。

However, we don’t want to push the train embeddings of each label to collapse into very small clusters. The only requirement is that given two positive examples of the same class and one negative example, the negative should be farther away than the positive by some margin. This is very similar to the margin used in SVMs, and here we want the clusters of each class to be separated by the margin.

> 但是，我们不想将每个标签的训练嵌入推到非常小的集群中。 唯一的要求是给定同一类的两个正例和一个负例，负例应该比正例更远一些。 这与 SVM 中使用的边距非常相似，这里我们希望每个类的集群都被边距分开。

To formalise this requirement, the loss will be defined over triplets of embeddings:

> 为了形式化这个要求，损失将在嵌入的三元组上定义：

* an anchor
* a positive of the same class as the anchor
* a negative of a different class

For some distance on the embedding space $d$, the loss of a triplet $(a, p, n)$

> 对于嵌入空间 $d$ 上的某个距离，三元组 $(a, p, n)$ 的损失

$$
L = \max(d(a, p) - d(a,n) + margin, 0)
$$

We minimize this loss, which pushes $d(a, p)$ to 0 and $d(a, n)$  to be greater than $d(a, p) + margin$. As soon as $n$ becomes an “easy negative”, the loss becomes zero.

> 我们最小化这个损失，它把 $d(a, p)$ 推到 0 并且 $d(a, n)$ 大于 $d(a, p) + margin$。 一旦 $n$ 变成“简单的负数”，损失就变为零。

## Triplet mining

Based on the definition of the loss, there are three categories of triplets:
* **easy triplets** triplets which have a loss of $0$, because $d(a, p) + margin < d(a, n)$
* **hard triplets** triplets where the negative is closer to the anchor than the positive, i.e. $d(a, n) < d(a, p)$
* **semi-hard triplets** triplets where the negative is not closer to the anchor than the positive, but which still have positive loss: $d(a, p) < d(a, n) < d(a, p) + margin$

> 根据损失的定义，三元组分为三类：
> * **简单的三元组** 损失 $0$ 的三元组，因为 $d(a, p) + margin < d(a, n)$
> * **硬三元组** 三元组，其中负样本比正样本更靠近锚点，即 $d(a, n) < d(a, p)$
> * **半硬三元组** 三元组，其中负数不比正数更靠近锚点，但仍然有正损失：$d(a, p) < d(a, n) < d(a, p ) + margin$

Each of these definitions depend on where the negative is, relatively to the anchor and positive. We can therefore extend these three categories to the negatives: **hard negatives**, **semi-hard negatives** or **easy negatives**.

> 这些定义中的每一个都取决于negative的位置，相对于锚点和正面。 因此，我们可以将这三个类别扩展到底片：**hard negatives**、**semi-hard negatives**或**easy negatives**。

The figure below shows the three corresponding regions of the embedding space for the negative.
> 下图展示了负样本嵌入空间的三个对应区域


> ![在这里插入图片描述](./imgs/triplets.png)
> The three types of negatives, given an anchor and a positive

Choosing what kind of triplets we want to train on will greatly impact our metrics. In the original Facenet paper, they pick a random semi-hard negative for every pair of anchor and positive, and train on these triplets.

> 选择我们想要训练什么样的三元组将极大地影响我们的指标。 在最初的 Facenet 论文中，他们为每一对锚和正样本选择一个随机的半硬负样本，并在这些三元组上进行训练。

## Offline and online triplet mining

We have defined a loss on triplets of embeddings, and have seen that some triplets are more useful than others. The question now is how to sample, or “mine” these triplets.

> 我们已经定义了嵌入三元组的损失，并且已经看到一些三元组比其他三元组更有用。 现在的问题是如何采样或“挖掘”这些三元组。

### Offline triplet mining

The first way to produce triplets is to find them offline, at the beginning of each epoch for instance. We compute all the embeddings on the training set, and then only select hard or semi-hard triplets. We can then train one epoch on these triplets.

> 产生三元组的第一种方法是离线找到它们，例如在每个 epoch 的开始。 我们计算训练集上的所有嵌入，然后只选择硬或半硬三元组。 然后我们可以在这些三元组上训练一个 epoch。

Concretely, we would produce a list of triplets $(i, j, k)$. We would then create batches of these triplets of size $B$, which means we will have to compute $3B$ embeddings to get the $B$ triplets, compute the loss of these $B$  triplets and then backpropagate into the network.

> 具体来说，我们将生成一个三元组 $(i, j, k)$ 的列表。 然后，我们将创建这些大小为 $B$ 的三元组的批次，这意味着我们必须计算 $3B$ 的嵌入来获得 $B$ 三元组，计算这些 $B$ 三元组的损失，然后反向传播到网络中。

Overall this technique is not very efficient since we need to do a full pass on the training set to generate triplets. It also requires to update the offline mined triplets regularly.

>  总的来说，这种技术不是很有效，因为我们需要对训练集进行一次完整的传递来生成三元组。 它还需要定期更新离线挖掘的三元组。

###  Online triplet mining

Online triplet mining was introduced in Facenet and has been well described by Brandon Amos in his blog post OpenFace 0.2.0: Higher accuracy and halved execution time.

> 在线三元组挖掘是在 Facenet 中引入的，Brandon Amos 在他的博客文章 OpenFace 0.2.0：更高的准确性和减半的执行时间中对此进行了很好的描述。

The idea here is to compute useful triplets on the fly, for each batch of inputs. Given a batch of $B$  examples (for instance $B$ images of faces), we compute the  $B$  embeddings and we then can find a maximum of $B^3$ triplets. Of course, most of these triplets are not valid (i.e. they don’t have 2 positives and 1 negative).

> 这里的想法是为每批输入动态计算有用的三元组。 给定一批 $B$ 示例（例如 $B$ 人脸图像），我们计算 $B$ 嵌入，然后我们可以找到最多 $B^3$ 三元组。 当然，这些三元组中的大多数都是无效的（即它们没有 2 个正数和 1 个负数）。

This technique gives you more triplets for a single batch of inputs, and doesn’t require any offline mining. It is therefore much more efficient. We will see an implementation of this in the last part.

> 这种技术为单批输入提供了更多的三元组，并且不需要任何离线挖掘。 因此，它的效率要高得多。 我们将在最后一部分看到它的实现。

> ![在这里插入图片描述](./imgs/online_triplet_loss.png)
> Triplet loss with online mining: triplets are computed on the fly from a batch of embeddings

## Strategies in online mining


In online mining, we have computed a batch of $B$ embeddings from a batch of $B$ inputs. Now we want to generate triplets from these $B$ embeddings.

> 在在线挖掘中，我们从一批 $B$ 输入中计算了一批 $B$ 嵌入。 现在我们想从这些 $B$ 嵌入中生成三元组。

Whenever we have three indices $i, j, k \in [1, B]$ , if examples $i$ and $j$  have the same label but are distinct, and example $k$  has a different label, we say that $(i, j, k)$  is a valid triplet. What remains here is to have a good strategy to pick triplets among the valid ones on which to compute the loss.

> 每当我们有三个索引 $i, j, k \in [1, B]$ 时，如果示例 $i$ 和 $j$ 具有相同的标签但不同，并且示例 $k$ 具有不同的标签，我们说 $(i, j, k)$ 是一个有效的三元组。 这里剩下的就是有一个好的策略来从计算损失的有效三元组中挑选三元组。

A detailed explanation of two of these strategies can be found in section 2 of the paper *In Defense of the Triplet Loss for Person Re-Identification.*

> 可以在论文的第 2 部分中找到对其中两种策略的详细解释 *In Defense of the Triplet Loss for Person Re-Identification*

They suppose that you have a batch of faces as input of size $B = PK$ , composed of $P$  different persons with $K$  images each. A typical value is $K = 4$. The two strategies are:

> 他们假设您有一批人脸作为输入，大小为 $B = PK$ ，由 $P$ 个不同的人组成，每个人都有 $K$ 个图像。 典型值为 $K = 4$。 这两种策略是：

* **batch all**: select all the valid triplets, and average the loss on the hard and semi-hard triplets.
	* a crucial point here is to not take into account the easy triplets (those with loss $0$), as averaging on them would make the overall loss very small
	* this produces a total of $PK(K − 1)(PK − K)$  triplets ($PK$  anchors, $K − 1$  possible positives per anchor, $PK − K$ possible negatives)

> * **batch all**：选择所有有效的三元组，平均硬和半硬三元组的损失。
>  	* 这里的一个关键点是不要考虑简单的三元组（那些损失 0 美元的），因为对它们进行平均会使整体损失非常小
>  	* 这会产生总共 $PK(K - 1)(PK - K)$ 三元组（$PK$ 锚点，$K - 1$ 每个锚点可能的正数，$PK - K$ 可能的负数）

* **batch hard**: for each anchor, select the hardest positive (biggest distance $d(a, p)$) and the hardest negative among the batch
	* this produces $PK$ triplets
	* the selected triplets are the hardest among the batch

> * **batch hard**：对于每个anchor，选择batch中最难的正例（最大距离$d(a, p)$）和最难的负例
> 		* 这会产生 $PK$ 三元组
> 		* 选择的三元组是批次中最难的

According to the paper cited above, the batch hard strategy yields the best performance:

*Additionally, the selected triplets can be considered moderate triplets, since they are the hardest within a small subset of the data, which is exactly what is best for learning with the triplet loss.*

> 根据上面引用的论文，batch hard 策略产生了最好的性能：
> *此外，选定的三元组可以被认为是中等三元组，因为它们在一小部分数据中是最难的，这正是使用三元组损失进行学习的最佳选择。*

However it really depends on your dataset and should be decided by comparing performance on the dev set.

> 但是，它实际上取决于您的数据集，应该通过比较开发集上的性能来决定。

----


# A naive implementation of triplet loss

In the stackoverflow answer, I gave a simple implementation of triplet loss for offline triplet mining:

> 在 stackoverflow 的回答中，我给出了一个简单的用于离线三元组挖掘的三元组损失实现：

```python
anchor_output = ...    # shape [None, 128]
positive_output = ...  # shape [None, 128]
negative_output = ...  # shape [None, 128]

d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

loss = tf.maximum(0.0, margin + d_pos - d_neg)
loss = tf.reduce_mean(loss)
```

The network is replicated three times (with shared weights) to produce the embeddings of $B$ anchors, $B$ positives and $B$ negatives. We then simply compute the triplet loss on these embeddings.

> 网络被复制 3 次（使用共享权重）以生成 $B$ 锚、$B$ 正数和 $B$ 负数的嵌入。 然后我们简单地计算这些嵌入的三元组损失。

This is an easy implementation, but also a very inefficient one because it uses offline triplet mining.

> 这是一个简单的实现，但也是一个非常低效的实现，因为它使用离线三元组挖掘。
----

# A better implementation with online triplet mining

All the relevant code is available on github in [model/triplet_loss.py](https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py).

> 所有相关代码都可以在 github 上的 [model/triplet_loss.py](https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py) 中找到。

There is an existing implementation of triplet loss with semi-hard online mining in TensorFlow: tf.contrib.losses.metric_learning.triplet_semihard_loss. Here we will not follow this implementation and start from scratch.

> 在 TensorFlow 中有一个使用半硬在线挖掘的三元组损失的现有实现：tf.contrib.losses.metric_learning.triplet_semihard_loss。 这里我们不会按照这个实现，从头开始。
> 
## Compute the distance matrix

As the final triplet loss depends on the distances $d(a, p)$  and $d(a, n)$ , we first need to efficiently compute the pairwise distance matrix. We implement this for the euclidean norm and the squared euclidean norm, in the _pairwise_distances function:

> 由于最终的三元组损失取决于距离 $d(a, p)$ 和 $d(a, n)$ ，我们首先需要有效地计算成对距离矩阵。 我们在 _pairwise_distances 函数中为欧几里得范数和平方欧几里得范数实现了这一点：

```python
def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances
```

To explain the code in more details, we compute the dot product between embeddings which will have shape $(B, B)$. The squared euclidean norm of each embedding is actually contained in the diagonal of this dot product so we extract it with tf.diag_part. Finally we compute the distance using the formula:

> 为了更详细地解释代码，我们计算嵌入之间的点积，其形状为 $(B, B)$。 每个嵌入的平方欧几里得范数实际上包含在这个点积的对角线上，所以我们用 tf.diag_part 提取它。 最后，我们使用公式计算距离：

$$
\|  a - b \|^2 = \| a \|^2 - 2(a, b) + \| b \|^2
$$

One tricky thing is that if squared=False, we take the square root of the distance matrix. First we have to ensure that the distance matrix is always positive. Some values could be negative because of small inaccuracies in computation. We just make sure that every negative value gets set to 0.0.

> 一件棘手的事情是，如果 squared=False，我们取距离矩阵的平方根。 首先，我们必须确保距离矩阵始终为正。 由于计算中的小不准确，一些值可能是负数。 我们只是确保每个负值都设置为 0.0。

The second thing to take care of is that if any element is exactly 0.0 (the diagonal should always be 0.0 for instance), as the derivative of the square root is infinite in 0, we will have a nan gradient. To handle this case, we replace values equal to 0.0 with a small $\epsilon = 1e^{-16}$. We then take the square root, and replace the values $\sqrt \varepsilon$ with 0.0.

> 第二件要注意的是，如果任何元素正好是 0.0（例如对角线应该总是 0.0），因为平方根的导数在 0 中是无限的，我们将有一个 nan 梯度。 为了处理这种情况，我们将等于 0.0 的值替换为小的 $\epsilon = 1e^{-16}$。 然后我们取平方根，并将值 $\sqrt\varepsilon$ 替换为 0.0。

## Batch all strategy

In this strategy, we want to compute the triplet loss on almost all triplets. In the TensorFlow graph, we want to create a 3D tensor of shape $(B, B, B)$  where the element at index $(i, j, k)$  contains the loss for triplet $(i, j, k)$.

> 在这个策略中，我们想要计算几乎所有三元组的三元组损失。 在 TensorFlow 图中，我们想要创建一个形状为 $(B, B, B)$ 的 3D 张量，其中索引 $(i, j, k)$ 处的元素包含三元组 $(i, j, k)$ 的损失。

We then get a 3D mask of the valid triplets with function _get_triplet_mask. Here, $mask[i, j, k]$ is true if $(i, j, k)$ is a valid triplet.

> 然后，我们使用函数 _get_triplet_mask 获得有效三元组的 3D 掩码。 在这里，如果 $(i, j, k)$ 是有效的三元组，则 $mask[i, j, k]$ 为真。

Finally, we set to $0$ the loss of the invalid triplets and take the average over the positive triplets.

> 最后，我们将无效三元组的损失设置为 $0$，并取正三元组的平均值。

Everything is implemented in function batch_all_triplet_loss:

> 一切都在函数 batch_all_triplet_loss 中实现：

```python
def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets
```

The implementation of _get_triplet_mask is straightforward, so I will not detail it.
> _get_triplet_mask 的实现很简单，我就不细说了。

## Batch hard strategy
In this strategy, we want to find the hardest positive and negative for each anchor.

> 在这个策略中，我们希望为每个锚点找到最难的正面和负面。

### Hardest positive

To compute the hardest positive, we begin with the pairwise distance matrix. We then get a 2D mask of the valid pairs $(a, p)$ (i.e. $a \neq p$  and $a$ and $p$ have same labels) and put to $0$ any element outside of the mask.

> 为了计算最难的正数，我们从成对距离矩阵开始。 然后我们得到有效对 $(a, p)$ 的 2D 掩码（即 $a \neq p$ 和 $a$ 和 $p$ 具有相同的标签）并将掩码外的任何元素放入 $0$。

The last step is just to take the maximum distance over each row of this modified distance matrix. The result should be a valid pair $(a,p)$  since invalid elements are set to $0$.

> 最后一步就是在这个修改后的距离矩阵的每一行上取最大距离。 结果应该是有效的对 $(a,p)$，因为无效元素设置为 $0$。

### Hardest negative
The hardest negative is similar but a bit trickier to compute. Here we need to get the minimum distance for each row, so we cannot set to $0$ the invalid pairs $(a,n)$  (invalid if $a$ and $n$ have the same label).

> 最难的负数是相似的，但计算起来有点棘手。 这里我们需要获取每行的最小距离，因此我们不能将无效对 $(a,n)$ 设置为 $0$（如果 $a$ 和 $n$ 具有相同的标签则无效）。

Our trick here is for each row to add the maximum value to the invalid pairs $(a,n)$ . We then take the minimum over each row. The result should be a valid pair $(a, n)$ since invalid elements are set to the maximum value.

> 我们这里的技巧是让每一行将最大值添加到无效对 $(a,n)$ 。 然后我们对每一行取最小值。 结果应该是一个有效的对 $(a, n)$ 因为无效元素被设置为最大值。

The final step is to combine these into the triplet loss:

> 最后一步是将这些组合成三元组损失：

```python
triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
```

Everything is implemented in function batch_hard_triplet_loss:
> 一切都在函数 batch_hard_triplet_loss 中实现：

```python
def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss
```

## Testing our implementation

If you don’t trust that the implementation above works as expected, then you’re right! The only way to make sure that there is no bug in the implementation is to write tests for every function in model/triplet_loss.py

> 如果您不相信上述实现按预期工作，那么您是对的！ 确保实现中没有错误的唯一方法是为 model/triplet_loss.py 中的每个函数编写测试

This is especially important for tricky functions like this that are difficult to implement in TensorFlow but much easier to write using three nested for loops in python for instance. The tests are written in model/tests/test_triplet_loss.py, and compare the result of our TensorFlow implementation with the results of a simple numpy implementation.

> 这对于像这样在 TensorFlow 中难以实现但在 python 中使用三个嵌套 for 循环更容易编写的棘手函数尤其重要。 测试写在 model/tests/test_triplet_loss.py 中，并将我们的 TensorFlow 实现的结果与一个简单的 numpy 实现的结果进行比较。

To check yourself that the tests pass, run:

> 要检查自己是否通过了测试，请运行：

```python
pytest model/tests/test_triplet_loss.py
```

(or just pytest)

Here is a list of the tests performed:

* test_pairwise_distances(): compare results of numpy of tensorflow for pairwise distance
* test_pairwise_distances_are_positive(): make sure that the resulting distance is positive
* test_gradients_pairwise_distances(): make sure that the gradients are not nan
* test_triplet_mask(): compare numpy and tensorflow implementations
* test_anchor_positive_triplet_mask(): compare numpy and tensorflow implementations
* test_anchor_negative_triplet_mask(): compare numpy and tensorflow implementations
* test_simple_batch_all_triplet_loss(): simple test where there is just one type of label
* test_batch_all_triplet_loss(): full test of batch all strategy (compares with numpy)
* test_batch_hard_triplet_loss(): full test of batch hard strategy (compares with numpy)

----

## Experience with MNIST

Even with the tests above, it is easy to oversee some mistakes. For instance, at first I implemented the pairwise distance without checking that the input to the square root was strictly greater than $0$. All the tests I had passed but the gradients during training were immediately nan. I therefore added test_gradients_pairwise_distances, and corrected the _pairwise_distances function.

> 即使进行了上述测试，也很容易监督一些错误。 例如，起初我在没有检查平方根的输入是否严格大于  $0$ 的情况下实现了成对距离。 我通过了所有测试，但训练期间的梯度立即为 nan。 因此，我添加了 test_gradients_pairwise_distances，并更正了 _pairwise_distances 函数。

To make things simple, we will test the triplet loss on MNIST. The code can be found here.

> 为简单起见，我们将在 MNIST 上测试三元组损失。代码可以在这里找到。

To train and evaluate the model, do:

> 要训​​练和评估模型，请执行以下操作：

```python
python train.py --model_dir experiments/base_model
```

This will launch a new experiment (i.e. a training run) named base_model. The model directory (containing weights, summaries…) is located in experiments/base_model. Here we use a json file experiments/base_model/params.json that specifies all the hyperparameters in the model. This file must be created for any new experiment.

> 这将启动一个名为 base_model 的新实验（即训练运行）。 模型目录（包含权重、摘要……）位于 Experiments/base_model。 这里我们使用一个json文件experiments/base_model/params.json来指定模型中的所有超参数。 必须为任何新实验创建此文件。

Once training is complete (or as soon as some weights are saved in the model directory), we can visualize the embeddings using TensorBoard. To do this, run:

> 一旦训练完成（或者在模型目录中保存了一些权重），我们就可以使用 TensorBoard 可视化嵌入。 为此，请运行：

```python
python visualize_embeddings.py --model_dir experiments/base_model
```

And run TensorBoard in the experiment directory:

> And run TensorBoard in the experiment directory:

```python
tensorboard --logdir experiments/base_model
```

These embeddings were run with the hyperparameters specified in the configuration file experiments/base_model/params.json. It’s pretty interesting to see which evaluation images get misclassified: a lot of them would surely be mistaken by humans too.

> 这些嵌入是使用配置文件experiments/base_model/params.json 中指定的超参数运行的。 看看哪些评估图像被错误分类是非常有趣的：其中很多肯定也会被人类误判。


![在这里插入图片描述](./imgs/embeddings.gif)


## Conclusion

TensorFlow doesn’t make it easy to implement triplet loss, but with a bit of effort we can build a good-looking version of triplet loss with online mining.

> TensorFlow 并不容易实现三元组损失，但通过一些努力，我们可以通过在线挖掘构建一个好看的三元组损失版本。

The tricky part is mostly how to compute efficiently the distances between embeddings, and how to mask out the invalid / easy triplets.

> 棘手的部分主要是如何有效地计算嵌入之间的距离，以及如何屏蔽无效/简单的三元组。

Finally if you need to remember one thing: always test your code, especially when it’s complex like triplet loss.

> 最后，如果您需要记住一件事：始终测试您的代码，尤其是当它像三元组丢失这样复杂时。

## Resources
Github repo for this blog post
Facenet paper introducing online triplet mining
Detailed explanation of online triplet mining in In Defense of the Triplet Loss for Person Re-Identification
Blog post by Brandon Amos on online triplet mining: OpenFace 0.2.0: Higher accuracy and halved execution time.
Source code for the built-in TensorFlow function for semi hard online mining triplet loss: tf.contrib.losses.metric_learning.triplet_semihard_loss.
The coursera lecture on triplet loss
