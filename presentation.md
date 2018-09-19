
name: inverse
layout: true
class: center, middle, inverse, animated, jackInTheBox

---
layout: false

class: center, middle
title: Support Vector Machines
subtitle: Subtitle
author: Darlan Cavalcante Moreira
date: 2018-09-24

.title[
{{title}}]

<!-- .subtitle[{{subtitle}}] -->

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

.titlepagebody[
.author[{{author}}]

.institution[GTEL]

.date[{{date}}]
]



.footnote[Created with [{Remark.js}](http://remarkjs.com/) using [{Markdown}](https://daringfireball.net/projects/markdown/) +  [{MathJax}](https://www.mathjax.org/)]


---
layout: true

.frametitle[
# Intuition
]

- Suppose we have a population of 50%-50% males and females
- We want to identify whether an individual is a Male or a Female
- **Features**: Individual height and individual and Hair Length

---

.center[<svg id="intuition_plot", height="430px"></svg>]

---

.center[<svg id="intuition_plot2", height="430px"></svg>]

---

.center[<svg id="intuition_plot3", height="430px"></svg>]

---

.center[<svg id="intuition_plot4", height="430px"></svg>]

---
layout: false

.frametitle[
# Intuition
]

- We call the samples closest to the frontier the "support vectors"

--

## Questions

--

- How can we find the support vectors?
--

- What is the best frontier?
--

- **Natural approach**: put the frontier as far away as possible from any support vector
--

<div class="moody box extra-top-bottom-margin">
<span class="label">Note:</span>
<p>This is the <strong>Maximal Margin Classifier</strong></p>
</div>
---

.frametitle[
# Maximal Margin Classifier
]


- Elegant and simple
--

- But cannot be applied to most datasets
  - Requires the classes to be separatable by a linear boundary
--

- The linear boundary is called an **hyperplane**

--

<div class="moody box extra-top-bottom-margin">
<span class="label">Note:</span>
<p>In a p-dimensional space, a hyperplane is a flat affine subspace of dimension p − 1</p>
</div>


---
layout: true

.frametitle[
# What is an Hyperplane
]

---

- Examples of Hyperplanes:
  - In two dimensions it is a flat one-dimensional subspace ➡ a line
$$\beta_0 + \beta_1 X_1 + \beta_2 X_2 = 0$$
--

  - In three dimensions it is a flat two-dimensional subspace ➡ a plane
--

  - In $p$ dimensions it is a flat $p-1$-dimensional subapce
$$\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p = 0 \tag{1}$$
--


<div class="moody box extra-top-bottom-margin">
<span class="label">Note:</span>
<p>If $X = (X_1, X_2, \ldots, X_p)^T$ and $(1)$ holds, then $X$ is a point on the hyperplane</p>
</div>


---

- The hyperplane as dividing p-dimensional space into two halves
--

- We can easily determine on which side of the hyperplane a point lies by
  calculating the sign of the left hand side of $(1)$

.center[<img style="height:400px"  src="figs/Hyperplane_1+2X_1+3X_2=0.png"><br/>Hyperplane for $1+2X_1+3X_2=0$]

--
<span class="figureannotation animated fadeInDown" style="top:35%; right:38%;">Region where $1+2X_1+3X_2 > 0$</span>
<span class="figureannotation animated fadeInUp" style="bottom:25%; right:38%;">Region where $1+2X_1+3X_2 < 0$</span>

---
layout: true

.frametitle[
# Classification
## Using a Separating Hyperplane
]

---

- Suppose that we have a $n \times p$ data matrix $\mtX$ that consists of $n$
training observations in $p$-dimensional space

  <math>$$\vtX_1 = \begin{bmatrix} x_{11} \\ \vdots \\ x_{1p}\end{bmatrix}, \ldots,\vtX_n=\begin{bmatrix}x_{n1}\\ \vdots \\ x_{np}\end{bmatrix}$$</math>

  and that these observations fall into two classes

- Suppose that it is possible to construct a hyperplane that separates the
training observations **perfectly** according to their class labels

???
- It is not always possible find a separating hyperplane that perfect separate
  the classes

---

### Examples of three separating hyperplanes


.footnote[We can label the observations as $y_i = 1$ and $y_i=-1$ for the blue and purple classes, respectively]

<!-- .center[<img src="figs/hyperplane_based_classifier.png" width="55%" style="margin-right: 30%">] -->

<!-- <div class="figureannotation animated fadeInRight sideannotation" style="top:45%;">A classifier based on a separating hyperplane leads to a linear decision boundary</div> -->


.center[
.captionbox[
![image](figs/hyperplane_based_classifier.png)
.caption[A classifier based on a separating hyperplane leads to a linear decision boundary]
]]


<!-- --- -->

<!-- - A separating hyperplane has the property that -->

<!--   $$\beta\_0 + \beta\_1 x\_{i1} + \beta\_2 x\_{i2}+\ldots+\beta\_p x\_{ip} > 0 \qquad\qquad\text{if } y = 1$$ -->
<!--   and -->
<!--   $$\beta\_0 + \beta\_1 x\_{i1} + \beta\_2 x\_{i2}+\ldots+\beta\_p x\_{ip} < 0 \qquad\qquad\text{if } y = -1$$ -->
<!-- -- -->

<!-- - We can rewrite this property as -->

<!--   $$y\_i(\beta\_0 + \beta\_1 x\_{i1} + \beta\_2 x\_{i2}+\ldots+\beta\_p x\_{ip}) > 0$$ -->

<!--   for all $i=1,\ldots,n$ -->

---

- We classify a test observation $x^*$ based on the sign of

  <math>$$f(x^*) = \beta_0 + \beta_1 x^*_{1} + \beta_2 x^*_{2}+\ldots+\beta_p x^*_{p}$$</math>
--

- If $f(x^*)$ is positive we assign the test observation to class $1$ and if it is negative we assign it to class $-1$
--


- We can also use the magnitude of $f(x^\*)$
  - If it is far from zero then $x^\*$ lies far from the hyperplane
  - We can be confident the class assignment of $x^\*$


---
layout: false

.frametitle[
# The Maximal Margin Classifier
]

- A natural choice for a hyperplane is the **maximal margin hyperplane**
  - The hyperplane that is farthest from the training observations
--

- We can compute the (perpendicular) distance from each training observation to
  a given separating hyperplane
--

- The smallest such distance is the minimal distance from the observations to
  the hyperplane
  - This distance it known as the **margin**
--

<div class="moody box">
<span class="label">Note:</span>
<p>The maximal margin hyperplane is the separating hyperplane for which the margin is largest</p>
</div>


--


- If we classify based on which side of the maximal margin hyperplane an
  observation lies, we get **Maximal Magin Classifier**

---
template: inverse

# Our Hope

A classifier with a large margin on the training data will also have a large margin on test data and hence it will classify the test observation correctly

---
layout: true
.frametitle[
# The Maximal Margin Classifier
]

- In the figure we see that the trainning observations are equidistant from the
  maximal margin hyperplane and lie along the dashed line

---

<!-- .center[<img src="figs/maximal_margin_classifier.png" width="50%" style="margin-right: 15%">] -->


<!-- <div class="figureannotation animated fadeInRight" style="top:45%; right:8%; width:13em">These three observations are known as the "support vectors"</div> -->

.center[.captionbox[
![image](figs/maximal_margin_classifier.png)
]]

---
count: false


.center[.captionbox[
![image](figs/maximal_margin_classifier.png)
.caption[.animated.fadeInRight[
These three observations are known as the "support vectors"]]
]]


???

- Notice the width of the margin
- The maximal margin hyperplane **only depend on the support vector**
- A movement to any of the other observations would not affect the separating
  hyperplane, provided that the observation’s movement does not cause it to
  cross the boundary set by the margin


---
layout: false
.frametitle[
# The Maximal Margin Classifier
## Construction
]

$$\begin{align}
& \underset{\beta\_0, \beta\_1, \ldots, \beta\_p, M}{\text{maximize}} & & M \tag{2a} \\\\
& \text{subject to} & & \sum\_{j=1}^p\beta\_j^2=1 \tag{2b}\\\\
& & & y\_i(\beta\_0 + \beta\_1 x\_{i1} + \beta\_2 x\_{i2} + \ldots + \beta\_p x\_{ip}) \geq M \tag{2c}
\end{align}$$
--

- $\text{(2c)}$ guarantees that each observation will be on the correct side of
the hyperplane, **provided that M is positive**
--

- $\text{(2b)}$ is not really a constraint on the hyperplane, but it adds meaning to $\text{(2c)}$
--

- The perpendicular distance from the $i$th observation to the hyperplane is given by

<math>$$y_i(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip})$$</math>

???

- Briefly, the maximal margin hyperplane is the solution to the optimization problem


---

.frametitle[
# The Non-separable Case
]

- In many cases a separating hyperplane does not exist

<!-- <div class="figureannotation" style="top:36%; right:13%; width:13em">We cannot *exactly* separate the two classes</div> -->
<!-- .center[<img src="figs/non-separable_case.png" width="45%" style="margin-right: 20%;">] -->

.center[
.captionbox[
![image](figs/non-separable_case.png)
.caption[We cannot *exactly* separate the two classes]
]]

???

- The maximum margin classifier is a very natural way to perform classification, **if a separating hyperplane exists**
- If an hyperplane does not exist then the optimization problem has no solution


--

- The previous optimization problem has no solution!

---
template: inverse

# Desire

Generalize the maximal margin classifier to the non-separable case ➡ Find a hyperplane that *almost* separates the classes


---
layout: true

.frametitle[
# Support Vector Classifier
## Motivation
]

---

- A classifier based on a separating hyperplane will necessarily perfectly
  classify all of the training observations
--

- In order to generalize the maximal margin classifier to the non-separable case
  we can use a so-called *soft margin*
--

  - We address this new classifier as "Support Vector Classifier"
--



- Even if a separating hyperplane does exists, there are cases where a
  classifier based on *hard margin* might not be desirable
--

  - A *hard margin* is too sensible to individual observations ➡ it may overfit the data
--

- By using a soft-margin we get:
 - Greater robustness to individual observations, and
 -  Better classification of most of the training observations.
---

- This hyperplane looks reasonable

<!-- .center[<img src="figs/maximal_margin_classifier_sensibility_1.png" width="55%" style="margin-right: 30%">] -->

<!-- <span class="figureannotation sideannotation" style="top:45%;">But what happens if we add a new training observation?</span> -->

.center[
.captionbox[
![image](figs/maximal_margin_classifier_sensibility_1.png)
.caption[But what happens if we add a new training observation?]
]]

---
layout: true
.frametitle[
# Support Vector Classifier
## Motivation
]

- 🤔, It does not look that nice anymore ➡ the margin is very thin

---

<!-- .center[<img src="figs/maximal_margin_classifier_sensibility_2.png" width="55%" style="margin-right: 30%">] -->

.center[
.captionbox[
![image](figs/maximal_margin_classifier_sensibility_2.png)
]]


---

count: false

.center[
.captionbox[
![image](figs/maximal_margin_classifier_sensibility_2.png)
.caption[.animated.fadeInRight[
Look how far a single new observation changed the hyperplane]]
]]

???

- The addition of a single observation leads to a **dramatic change** in the
maximal margin hyperplane.
- The resulting maximal margin hyperplane is not satisfactory: **for one thing,
it has only a tiny margin**
- This is problematic because as discussed previously, the distance of an
observation from the hyperplane can be seen as a measure of our confidence that
the obser- vation was correctly classified.

---
template: inverse

# Insight

It is worthwhile to misclassify a few training observations in
order to do a better job in classifying the remaining observations

---
layout: false
.frametitle[
# Support Vector Classifier
## Overview
]

- The support vector classifier allow some observations to be on the incorrect
  side of the margin
--

  - or even on the incorrect side of the hyperplane
--

- Observations on the wrong side of the hyperplane correspond to training
  observations that are misclassified by the classifier
--


<div class="angry box extra-top-bottom-margin">
<span class="label">Note:</span>
<p>When there is no separating hyperplane, such a situation is inevitable</p>
</div>

---

layout: true
.frametitle[
# Support Vector Classifier
## Example
]

---

- A support vector classifier was fit to a small data set

.center[
.captionbox[
![image](figs/support_vector_classifier_1.png)
]]

---
count: false

- A support vector classifier was fit to a small data set

.center[
.captionbox[
![image](figs/support_vector_classifier_1.png)
.caption[.animated.fadeInRight[
We have points on the wrong side of the margin]]
]]

<!-- <span class="figureannotation animated fadeInRight" style="top:45%; right:8%; -->
<!-- width:13em">We have points on the wrong side of the margin</span> -->

???

- No observations are on the wrong side of the hyperplane.
- We could have found an hyperplane and a margin size where all points would be
  on the right side of the margin

---

- Add two additional points, $11$ and $12$

<!-- .center[<img src="figs/support_vector_classifier_2.png" width="60%" style="margin-right: 30%;">] -->

.center[
.captionbox[
![image](figs/support_vector_classifier_2.png)
]]

---
count: false

- Add two additional points, $11$ and $12$

.center[
.captionbox[
![image](figs/support_vector_classifier_2.png)
.caption[.animated.fadeInRight[
Finding a perfect separating hyperplane is impossible here even if we wanted to]]
]]

<!-- <span class="figureannotation animated fadeInRight" style="top:45%; right:8%; -->
<!-- width:13em">Finding a perfect separating hyperplane is impossible here even if we wanted to</span> -->

???

These two observations are on the wrong side of the hyperplane and the wrong
side of the margin.

---
layout: true

.frametitle[
# Support Vector Classifier
## Details of the Support Vector Classifier
]

---

- Finding the hyperplane of the support vector classifier is the solution to the
  optimization problem

  <math style="font-size: 80%; margin-left: -2.5em;">$$\begin{align}
    & \underset{\beta_0, \beta_1, \ldots, \beta_p, \class{alert}{\epsilon_1, \ldots, \epsilon_n,} M}{\text{maximize}} & & M \tag{3a}\\
    & \text{subject to} & & \sum_{j=1}^p\beta_j^2=1 \tag{3b}\\
    & & & y_i(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip}) \geq M\class{alert}{(1-\epsilon_i)} \tag{3c} \\
    & & & \class{alert}{\epsilon_i \geq 0, \sum_{i=1}^n\epsilon_i \leq C} \tag{3d} \end{align}$$</math>

  where $C$ is a nonnegative tuning parameter

???

Notice the changes from the previous optimization problem!

---

- $M$ is still the width of the margin and we seek to make this quantity as
large as possible.
- $\epsilon_1, \ldots, \epsilon_n$ are slack variables that allow individual
observations to be on the wrong side of the margin or the hyperplane
- Once we have solved (3a)–(3d), we classify a test observation $x^*$ as before,
  by simply determining on which side of the hyperplane it lies

---
template: inverse

# Interpretation

The slack variable $\epsilon_i$ tells us where the $i$th observation is located
relative to the hyperplane and relative to the margin

---

layout: true

.frametitle[
# Support Vector Classifier
## Details of the Support Vector Classifier
]

---

- If $\epsilon_i = 0$ then the $i$th observation is on the correct side of the margin
--


- If $\epsilon_i > 0$ then the $i$th observation is on the wrong side of the margin
  - The $i$th observation has violated the margin
--


- If $\epsilon_i > 1$ then the $i$th observation is on the wrong side of the hyperplane

---

template: inverse

And what about $C$?


---

- $C$ bounds the sum of the $\epsilon_i$'s ➡ it determines the number and
  severity of the violations to the margin (and to the hyperplane) that we
  tolerate
--

- We can think of $C$ as a budget for the amount that the margin can be violated by the $n$ observations
--

  - If $C = 0$ then there is no budget for violations to the margin ➡ all
    $\epsilon_i$ values are equal to zero
  - For $C > 0$ no more than $C$ observations can be on the wrong side of the hyperplane
--

- As the budget C increases, we become more tolerant of
violations to the margin ➡ the margin will widen

---

template: inverse


# Interpretation

$C$ controls the bias-variance trade-off


It is treated as a tuning parameter that is generally **chosen via
cross-validation**



---
layout: false
.frametitle[
# Support Vector Classifier
## Interesting properties of $(\text{3a})$-$(\text{3d})$
]

- Only observations that either lie on the margin or that violate the margin
  will affect the hyperplane
--

- Observations that lie directly on the margin, or on the wrong side of the
  margin for their class, are known as support vectors.
--


<div class="moody box extra-top-bottom-margin">
<span class="label">Note:</span>
<p>When the tuning parameter $C$ is large the margin is wide and many observations violate the margin ➡ there are many support vectors</p>
</div>


---
layout: true

.frametitle[
# Support Vector Classifier
## Effect of varying C
]

- As we decrease the value of $C$ the margin narrows and we have fewer support vectors

---

.center[
.captionbox[
![image](figs/support_vector_classifier_vary_C_1.png)
.caption[.animated.fadeInRight[
Many observations violate the margin, and so there are many support
vectors]]
]]

<!-- .center[<img src="figs/support_vector_classifier_vary_C_1.png" width="60%" style="padding-right: 10em">] -->


<!-- <span class="figureannotation animated fadeInDown" style="top:50%; right:8%; -->
<!-- width:13em">Many observations violate the margin, and so there are many support -->
<!-- vectors</span> -->



---

.center[
.captionbox[
![image](figs/support_vector_classifier_vary_C_2.png)
.caption[.animated.fadeInRight[
As $C$ decreases the margin decreases and we have fewer support vectors]]
]]

<!-- .center[<img src="figs/support_vector_classifier_vary_C_2.png" width="60%" style="padding-right: 10em">] -->


<!-- <span class="figureannotation animated fadeInDown" style="top:50%; right:8%; width:13em">As $C$ decreases the margin decreases and we have fewer support vectors</span> -->

---

.center[
.captionbox[
![image](figs/support_vector_classifier_vary_C_3.png)
.caption[.animated.fadeInRight[
This means we reduce the bias, but since individual observations have a larger impact we are increasing the variance]]
]]

<!-- .center[<img src="figs/support_vector_classifier_vary_C_3.png" width="60%" style="padding-right: 10em">] -->


<!-- <span class="figureannotation animated fadeInDown" style="top:50%; right:8%; width:13em">This means we reduce the bias, but since individual observations have a larger impact we are increasing the variance</span> -->

---

.center[
.captionbox[
![image](figs/support_vector_classifier_vary_C_4.png)
.caption[.animated.fadeInRight[
If we keep decreasing $C$ we walk towards the maximal margin classifier, but for this dataset there is no solution with $C=0$]]
]]

<!-- .center[<img src="figs/support_vector_classifier_vary_C_4.png" width="60%" style="padding-right: 10em">] -->


<!-- <span class="figureannotation animated fadeInDown" style="top:50%; right:8%; width:13em">If we keep decreasing $C$ we walk towards the maximal margin classifier, but for this dataset there is no solution with $C=0$</span> -->

---
template: inverse

# Insight

The fact that the support vector classifier's decision rule is based only on a
potentially small subset of the training observations (the support vectors)
means that it is quite robust to the behavior of observations that are far away
from the hyperplane


---
layout: true

.frametitle[
# Non-linear Decision Boundaries
]

---

- In practice we are sometimes faced with non-linear class boundaries
- A linear decision boundary is useless in this case

.center[<img src="figs/support_vector_machine_linear_bad.png" width="430px">]


---

- One alternative to address this non-linearity is to enlarge the feature space
  ➡ using functions of the predictors such as quadratic and cubic terms
--

- Rather than fitting a support vector classifier using $p$ features
<math>$$X_1, X_2, \ldots, X_p$$</math>
 we could instead fit a support vector classifier using $2p$ features
<math>$$X_1, X_1^2, X_2, X_2^2, \ldots, X_p, X_p^2$$</math>
--

- Other functions of the predictors could be considered rather than polynomials


<div class="moody box">
<span class="label">Note:</span>
<p>There are many possible ways to enlarge the feature space<br>Is there a better way?</p>
</div>

---

template: inverse

![kernels](figs/kernels.jpg)

???

- Kernels are a simple and efficient way to **implicitly** enlarge the feature space



---
layout: true

.frametitle[
# Support Vector Machines]

---

- SVM is an extension of the support vector classifier that results from
  implicitly enlarging the feature space using a *kernel trick*
--


<div class="happy box extra-top-bottom-margin">
<span class="label">Note:</span>
<p>But first we need to rewrite the optimization problem in a more general way</p>
</div>

---

- The **solution** of the vector classifier problem $(\text{3a})–(\text{3d})$ involves
  only inner products of the observations
- The inner product of two observations $x\_i$, $x\_i'$ is given by
  $$\langle x\_i, x\_i' \rangle = \sum\_{j=1}^p x\_{ij}x\_{ij}'$$
--

- The linear support vector classifier can be represented as
  $$f(x) = \beta\_0 + \sum\_{i=1}^{n}\alpha\_i \langle x\_i, x\_i' \rangle \tag{4}$$
  where there are $n$ parameters $\alpha\_i$, $i=1,\ldots, n$

---

- In order to evaluate the function $f(x)$ in $(4)$ we need to
compute the inner product between the new point $x$ and each of the training
points $x_i$
--

- But $\alpha_i$ is **nonzero only for the support vectors** in the solution
--

- We can then change the previous Equation to

  <math>$$f(x) = \beta_0 + \sum_{i \in \stS}\alpha_i \langle x_i, x_i' \rangle \tag{5}$$</math>

  which typically involves far fewer terms than $(4)$.
--

<div class="happy box">
<span class="label">Note:</span>
<p>To summarize, in representing the linear classifier $f(x)$, and in computing its coefficients, all we need are inner products</p>
</div>


---

- Now let's replace the inner product with a *generalization* in the form
  $$K(x\_i,x\_i')$$

  where $K$ is some function that we will refer as the *kernel*
--


<div class="moody box">
<span class="label">Note:</span>
<p>A kernel is a function that quantifies the similarity of two observations</p>
</div>

--


- **Example**: Linear Kernel (we get the support vector classifier)

<math>$$K(x_i, x_i') = \langle x_{i}, x_i'\rangle = \sum_{j=1}^p x_{ij} x_{i'j} \tag{6}$$</math>

---

layout: true

.frametitle[
# Support Vector Machines
## Other types of kernel]

---

### Polynomial kernel of degree $d$

$$K(x\_i,x\_i') = \left( 1 + \sum\_{j=1}^p x\_{ij} x\_{i'j} \right)^d$$
--

- Using such a kernel with $d > 1$, instead of the standard linear kernel leads
to a much more flexible decision boundary

Note that in this case the (non-linear)
function has the form

<math>$$f(x) = \beta_0 \sum_{i \in \stS}\alpha_i K(x, x_i)$$</math>

???

- It essentially amounts to fitting a support vector classifier in a
higher-dimensional space involving polynomials of degree d, rather than in the
original feature space.

---

- An SVM with a polynomial kernel of degree 3 applied to the non-linear data from before

.center[
.captionbox[
![image](figs/polynomial_kernel_degree_3.png)
.caption[.animated.fadeInRight[
The fit is a substantial improvement over the linear support vector classifier]]
]]


<!-- .center[<img src="figs/polynomial_kernel_degree_3.png" width="55%" style="padding-right:10em;">] -->

<!-- <span class="figureannotation animated fadeInRight" style="top:50%; right:8%; width:13em">The fit is a substantial improvement over the linear support vector classifier</span> -->

---

## Radial Kernel

$$K(x\_i, x\_i') = \exp \left( -\gamma \sum\_{j=1}^p (x\_{ij}-x'\_{ij})^2 \right)= \exp\left( -\gamma \Vert x\_i - x\_i' \Vert^2 \right)$$

--


&nbsp;
<div class="moody box">
<span class="label">Note:</span>
<p>If a test observation $x^*$ is far from a training observation $x_i$ in terms of Euclidian distance, then $K(x_i, x_i')$ is very tiny</p>
</div>


- $x_i$ essentially takes no role in predicting the label for $x^*$

---

- An SVM with a radial kernel applied to the non-linear data from before

.center[
.captionbox[
![image](figs/radial_kernel.png)
.caption[.animated.fadeInRight[
The radial kernel also performs much better than the linear kernel]]
]]

<!-- .center[<img src="figs/radial_kernel.png" width="55%" style="padding-right:10em;">] -->

<!-- <span class="figureannotation animated fadeInRight" style="top:50%; right:8%; width:13em">The radial kernel also performs much better than the linear kernel</span> -->

---

template: inverse


When the support vector classifier is combined with a non-linear kernel the
resulting classifier is known as a **support vector machine**


---
layout: false
.frametitle[
# Support Vector Machines
## Why use kernels]


- What is the advantage of using a kernel rather then enlarging the feature
  space using functions of the original features?
--

  - Computational: We only need to compute $K(x\_i,x\_i')$ for $\binom{n}{2}$
    distinct pairs
  - Which can be done **in the original feature space**
--


<div class="angry box extra-top-bottom-margin">
<span class="label">Note:</span>
<p>In many applications of SVMs, the enlarged feature space is so large that computations are intractable
<ul>
<li>The radial kernel has an implicit infinite-dimensional feature space!</li>
</ul>

</p>
</div>

---
template: inverse

# The kernel trick is awesome!


But SVM is still limited two separating between two classes, right?


---
.frametitle[
# SVMs with $K>2$ classes
]

- The concept of separating hyperplanes does not lend itself naturally to more
  than two classes

???
- Some proposed extensions to the $K$-class case are:

--

### one-versus-one

- Construct $\binom{K}{2}$ SVMs, each comparing a pair of classes
- The final classification is performed by assigning the test observation to the
  class to which it was most frequently assigned in these $\binom{K}{2}$
  pairwise classifications
--


### one-versus-all

- Fit $K$ SVMs, each time comparing one of the $K$ classes to the remaining
  $K-1$ classes
- Assign the observation to the class for which $\beta\_{0k}+\beta\_{1k}x\_1^\*+\beta\_{2k}x\_2^\*+\ldots+\beta\_{pk}x\_p^\*$ is largest ➡ hightest confidence

---

template: inverse

## Meh, but SVM only works in classification problems

---

.frametitle[
# Regression with SVMs
]

.extra-top-bottom-margin[
### Not on this presentation
]

<div class="happy box">
<span class="label">More Info:</span>
<p>See
<br><a href="https://alex.smola.org/papers/2003/SmoSch03b.pdf">https://alex.smola.org/papers/2003/SmoSch03b.pdf</a>
<br><a href="http://kernelsvm.tripod.com/">http://kernelsvm.tripod.com/</a>
<br><a href="https://en.wikipedia.org/wiki/Support_vector_machine?oldformat=true">https://en.wikipedia.org/wiki/Support_vector_machine?oldformat=true</a>
<br><a href="https://sadanand-singh.github.io/posts/svmmodels/">https://sadanand-singh.github.io/posts/svmmodels/</a>
</p>
</div>

---
background-image: url(figs/the_end.png)
background-size: cover
