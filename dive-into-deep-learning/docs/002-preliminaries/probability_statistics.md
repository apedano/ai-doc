# Probability and statistics

https://d2l.ai/chapter_preliminaries/probability.html

In **supervised learning**, we want to predict something unknown (the _target_) given something known (the _features_). 
Depending on our objective, we might

* **attempt to predict the most likely value of the target**. 
* Or we might predict the **value with the smallest expected distance from the target**.
* And sometimes we wish not only to predict a specific value but to **quantify our uncertainty**. 

For example, given some features describing a patient, we might want to know how likely they are to suffer a heart attack 
in the next year. 

In **unsupervised learning**, we often care about uncertainty. 
To determine whether a set of measurements are anomalous, it helps to know how likely one is to observe values in a population of interest.

Furthermore, in **reinforcement learning**, we wish to develop agents that act intelligently in various environments. 
This requires reasoning about **how an environment might be expected to change** and what **rewards one might expect to encounter in response to each of the available actions**.

> <span style="color:red">Probability<span/> is the mathematical field concerned with reasoning under uncertainty. 
 
Given a probabilistic model of some process, we can reason about the likelihood of various events.

## Bayesian probability

The use of probabilities to **describe the frequencies of repeatable events** (like coin tosses) is fairly uncontroversial. 

In fact, <span style="color:red">frequentist scholars adhere to an interpretation of probability that applies only to such repeatable events<span style="color:red">. 

By contrast <span style="color:red">Bayesian scholars use the language of probability more broadly to formalize reasoning under uncertainty<span/>. 

<span style="color:red">Bayesian probability <span/> is characterized by two unique features: 
* (i) assigning degrees of **belief to non-repeatable events**, e.g., what is the probability that a dam will collapse?; and 
* (ii) **subjectivity**. 

>While Bayesian probability provides unambiguous rules for how one should **update their beliefs in light of new evidence**, 
it allows for different individuals to **start off with different prior beliefs**. 


## Statistics 

>Statistics helps us to reason **backwards**, starting off with collection and organization of data and backing out to 
what inferences we might draw about the process that generated the data. 

Whenever we analyze a dataset, hunting for patterns that we hope might characterize a broader population, 
we are employing statistical thinking.  

## A Simple Example: Tossing Coins

Imagine that we plan to toss a coin and want to quantify how likely we are to see heads (vs. tails).
If the coin is *fair*, then both outcomes (heads and tails), are equally likely.

Moreover, if we plan to toss the coin $n$ times then the fraction of heads that we *expect* to see
should exactly match the *expected* fraction of tails.

One intuitive way to see this is by symmetry:
for every possible outcome with 

$n_\textrm{h}$ heads and $n_\textrm{t} = (n - n_\textrm{h})$ tails,

there is an equally likely outcome with $n_\textrm{t}$ heads and $n_\textrm{h}$ tails.

Note that this is only possible if on average we expect to see $1/2$ of tosses come up heads
and $1/2$ come up tails.

Of course, if you conduct this experiment many times with $n=1000000$ tosses each,
you might never see a trial where $n_\textrm{h} = n_\textrm{t}$ exactly.


> Formally, the quantity $1/2$ is called a <span style="color:red">*probability*</span>
> and here it captures the certainty with which any given toss will come up heads.
Probabilities assign scores between $0$ and $1$ to outcomes of interest, called <span style="color:red">*events*</span>.

> So the probabiliy of the event $\textrm{heads}$ is $P(\textrm{heads}) \in [0,1]$ 


A probability of $1$ indicates absolute certainty (imagine a trick coin where both sides were heads)
and a probability of $0$ indicates impossibility (e.g., if both sides were tails).

> The frequencies $n_\textrm{h}/n$ and $n_\textrm{t}/n$ are not probabilities but rather <span style="color:red">*statistics*</span>.

<span style="color:red">Probabilities are *theoretical* quantities</span> that underly the data generating process. Here, the probability $1/2$
is a property of the coin itself.

By contrast, <span style="color:red">statistics are *empirical* quantities </span> that are computed as functions of the observed data.

### Estimators
Our interests in probabilistic and statistical quantities are inextricably intertwined. 

>We often design special statistics called <span style="color:red">*estimators*</span> that, given a dataset, produce *estimates*
of model parameters such as probabilities.

> Moreover, when those estimators satisfy a nice property called <span style="color:red">*consistency*</span>, our estimates will converge
to the corresponding probability.

In turn, these inferred probabilities tell about the likely statistical properties of data from the same population
that we might encounter in the future.

Suppose that we stumbled upon a real coin for which we did not know the true $P(\textrm{heads})$.
To investigate this quantity with statistical methods, we need to 
* (i) collect some data (<span style="color:red">*sampling*</span>);
* (ii) design an estimator.


Data acquisition here is easy; we can toss the coin many times and record all the outcomes.
Formally, drawing realizations from some underlying random process is called .

As you might have guessed, one natural estimator is the ratio of the number of observed *heads*
to the total number of tosses.

Now, suppose that the coin was in fact fair, i.e., $P(\textrm{heads}) = 0.5$.

To simulate tosses of a fair coin, we can invoke any random number generator.
There are some easy ways to draw samples of an event with probability $0.5$.

For example Python's `random.random` yields numbers in the interval $[0,1]$
where the probability of lying in any sub-interval $[a, b] \subset [0,1]$ is equal to $b-a$, 
for instance ($P(0.2≤X≤0.5)=0.5−0.2=0.3$).

Thus, we can get out `0` and `1` with probability `0.5` each by testing whether the returned float number is greater than `0.5`:

```python
import random
def get_statistics_coin(num_experiments):
    num_heads = sum([random.random() > 0.5 for _ in range(num_experiments)])
    return num_heads, num_experiments - num_heads

print("heads, tails: ", get_statistics_coin(100)) #heads, tails:  (56, 44)
print("heads, tails: ", get_statistics_coin(10000)) #heads, tails:  (5042, 4958)
print("heads, tails: ", get_statistics_coin(100000)) #heads, tails:  (49794, 50206)
```

### The multinomial function
More generally, we can simulate multiple draws from any variable with a finite number 
of possible outcomes (like the toss of a coin or roll of a die)
by calling the multinomial function, setting the first argument
to the number of draws and the second as a list of probabilities
associated with each of the possible outcomes.

To simulate ten tosses of a fair coin, we assign probability vector `[0.5, 0.5]`=$[P(heads), P(tails)]$.

The function returns a vector with length equal to the number of possible outcomes (here, 2),
where the first component tells us the number of occurrences of heads and the second component tells us
the number of occurrences of tails.

```python
from torch.distributions import Multinomial
import torch
fair_probs = torch.tensor([0.5, 0.5]) 
Multinomial(100, fair_probs).sample() #Can vary ex. tensor([41., 59.])
```
To get the frequency (statistics)

```python
num_samples = 1000
freqency = Multinomial(num_samples, fair_probs).sample()/num_samples
print(freqency) #tensor([0.4980, 0.5020])
```

### The law of the large numbers

> In general, for averages of repeated events (like coin tosses), as the number of repetitions grows, 
our estimates are guaranteed to converge to the true underlying probabilities. 
 
>The mathematical formulation of this phenomenon is called the <span style="color:red">**law of large numbers**</span> 
and the **central limit theorem** tells us that in many situations, as the sample size  $n$  grows, 
these errors should go down at a rate of  $(1/\sqrt n)$ . 

Let's get some more intuition by studying how our estimate evolves as we grow the number of tosses from 1 to 10,000.

```python
from torch.distributions import Multinomial
import torch
#estimated probs of head and tail
fair_probs = torch.tensor([0.5, 0.5])
#each sample chooses exactly one category and we repeat it 10000 times
#Possible outputs:
#    tensor([1., 0.])   # heads
#    tensor([0., 1.])   # tails
#So counts has shape: (10000, 2)
# head:
#   tensor([
#    [1., 0.],
#    [0., 1.],
#    [1., 0.],
#    ...
#   ])
counts = Multinomial(1, fair_probs).sample((10000,))
#Cumulative counts on dim=0: computes running totals down the rows
# so with the previus example is
#   [1,0]
#   [1,1]
#   [2,1]
cum_counts = counts.cumsum(dim=0)
#Convert counts into probability estimates
#estimate=(cumulative counts)/(total flips so far)
# if cumulative count = [2,1] then total flip is 3
#and estimates becomes [2/3,1/3]
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
#Converts the tensor into a NumPy array for easier plotting or analysis.
estimates = estimates.numpy()
```
Now we can plot the numpy array

```python
import matplotlib.pyplot as plt
plt.plot(estimates[:, 0], label=("P(coin=heads)"))
plt.plot(estimates[:, 1], label=("P(coin=tails)"))
plt.axhline(y=0.5, color='red', linestyle='dashed')

plt.gca().set_xlabel('Samples')
plt.gca().set_xlabel('Samples')
plt.legend()
plt.show()
```

![plot_coin_estimates.png](img/plot_coin_estimates.png)

The dashed red line gives the true underlying probability. 
As we get more data by conducting more experiments, the curves converge towards the true probability. 

### A more formal treatment

#### Sample space or outcome space $\mathcal{S}$

> The set of possible outcomes where each element is a distinct possible *outcome*.

In the case of rolling a single coin,
$\mathcal{S} = \{\textrm{heads}, \textrm{tails}\}$.

For a single die, $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$.


When flipping two coins, possible outcomes are $\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails}), (\textrm{tails}, \textrm{heads}),  (\textrm{tails}, \textrm{tails})\}$.


### Events 

> *Events* are subsets of the sample space ("_seeing a $5$_" $\mathcal{A}$, "_seeing an odd number_" $\mathcal{B}$). 

For instance, the event "_the first coin toss comes up heads_"
corresponds to the set $\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails})\}$.

Whenever the outcome $z$ of a random experiment satisfies $z \in \mathcal{A}$, then event $\mathcal{A}$ has occurred.

For a single roll of a die, we could define the events "_seeing a $5$_" ($\mathcal{A} = \{5\}$)
and "_seeing an odd number_"  ($\mathcal{B} = \{1, 3, 5\}$). 
In this case, if the die came up $5$, we would say that both $\mathcal{A}$ and $\mathcal{B}$ occurred.
On the other hand, if $z = 3$, then $\mathcal{A}$ did not occur but $\mathcal{B}$ did.

### Probability functions

> A *probability* function maps events onto real values ${P: \mathcal{A} \subseteq \mathcal{S} \rightarrow [0,1]}$.
The probability, denoted $P(\mathcal{A})$, of an event $\mathcal{A}$
in the given sample space $\mathcal{S}$
has the following properties:

#### Probability properties
* The probability of any event $\mathcal{A}$ is a nonnegative real number, i.e., <span style="color:red">$P(\mathcal{A}) \geq 0$</span>;
* The probability of the entire sample space is $1$, i.e., <span style="color:red">$P(\mathcal{S}) = 1$</span>;
* For any countable sequence of events $\mathcal{A}_1, \mathcal{A}_2, \ldots$ that are *mutually exclusive* (i.e., $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ for all $i \neq j$), the probability that any of them happens is equal to the sum of their individual probabilities, i.e., <span style="color:red">$P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$</span>.

These axioms of probability theory, proposed by :citet:`Kolmogorov.1933`, can be applied to rapidly derive a number of important consequences.

For instance, it follows immediately that the probability of any event $\mathcal{A}$
*or* its complement $\mathcal{A}'$ occurring is 1 (because <span style="color:red">$\mathcal{A} \cup \mathcal{A}' = \mathcal{S}$</span>).

We can also prove that $P(\emptyset) = 0$ because: 

$1 = P(\mathcal{S} \cup \mathcal{S}') = P(\mathcal{S} \cup \emptyset) = P(\mathcal{S}) + P(\emptyset) = 1 + P(\emptyset)$.

Consequently, the probability of any event $\mathcal{A}$ *and* its complement $\mathcal{A}'$ occurring simultaneously is $P(\mathcal{A} \cap \mathcal{A}') = 0$.
Informally, this tells us that impossible events have zero probability of occurring.

## Random variables

A random variable is a function that assigns a numerical value to each outcome of a random experiment.

$$
X \in \mathcal{S} → \mathbb{R}
$$

For instance, the dice roll has 

$$
\mathcal{S} = {1,2,3,4,5,6}
$$

Two examples or random variables are

* $X(w)=w$ the random variable equals the number rolled.
* $Y = \left\{ \begin{array}{rcl}
1 & \mbox{if roll is even} \\ 0 & \mbox{if roll is odd} 
\end{array}\right.$

Every value taken by a random variable corresponds
to a subset of the underlying sample space.
Thus, the occurrence where the random variable $X$
takes value $v$, denoted by $X=v$, is an *event*
and $P(X=v)$ denotes its probability.

TO BE FINISHED

Sometimes this notation can get clunky,
and we can abuse notation when the context is clear.
For example, we might use $P(X)$ to refer broadly
to the *distribution* of $X$, i.e.,
the function that tells us the probability
that $X$ takes any given value.
Other times we write expressions
like $P(X,Y) = P(X) P(Y)$,
as a shorthand to express a statement
that is true for all of the values
that the random variables $X$ and $Y$ can take, i.e.,
for all $i,j$ it holds that $P(X=i \textrm{ and } Y=j) = P(X=i)P(Y=j)$.
Other times, we abuse notation by writing
$P(v)$ when the random variable is clear from the context.
Since an event in probability theory is a set of outcomes from the sample space,
we can specify a range of values for a random variable to take.
For example, $P(1 \leq X \leq 3)$ denotes the probability of the event $\{1 \leq X \leq 3\}$.


Note that there is a subtle difference
between *discrete* random variables,
like flips of a coin or tosses of a die,
and *continuous* ones,
like the weight and the height of a person
sampled at random from the population.
In this case we seldom really care about
someone's exact height.
Moreover, if we took precise enough measurements,
we would find that no two people on the planet
have the exact same height.
In fact, with fine enough measurements,
you would never have the same height
when you wake up and when you go to sleep.
There is little point in asking about
the exact probability that someone
is 1.801392782910287192 meters tall.
Instead, we typically care more about being able to say
whether someone's height falls into a given interval,
say between 1.79 and 1.81 meters.
In these cases we work with probability *densities*.
The height of exactly 1.80 meters
has no probability, but nonzero density.
To work out the probability assigned to an interval,
we must take an *integral* of the density
over that interval.

