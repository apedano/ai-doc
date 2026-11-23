# Reinforcement Learning for LLMs

https://cameronrwolfe.substack.com/p/llm-rl

https://chatgpt.com/c/6acba816-3ee0-83eb-9b62-837e400b7c23

https://cameronrwolfe.substack.com/p/agentic-rl

Reinforcement learning (RL) has played a crucial role throughout the history of research on large language models (LLMs)
Many of the most pressing problems we are trying to solve _today—reasoning_, _knowledge work_, _agents_, _token
efficiency_, _reliability_,
and more are being addressed through RL.

In this overview, we will work up to the frontier of RL research by starting from first principles.

## General framework for RL

A formal structure for RL, outlining the learning objective used for RL training.

### Actions

> When running RL training, we have an <span style="color:red">**agent**</span> that, at every time step $t$ iteratively
> takes <span style="color:red">**actions $a_t$**</span> within some <span style="color:red">**environment**</span>;
 

### Policy

> The actions are predicted by a <span style="color:red">**policy**</span> <span style="color:red">**$\pi_{\Theta}$
**</span> (we can think of this as the agent’s brain)
> with parameters $\Theta$ (e.g., the policy is the LLM itself in the context of training an LLM).

Our policy can either be <u>deterministic</u> or <u>stochastic</u>, but for <span style="color:#FF0000">**LLMs**</span> we will assume the policy
is <span style="color:red">**stochastic**</span>.

Our stochastic policy $\pi_{\Theta}$ outputs a probability distribution over the set of possible actions,
and an action $a_t$ is sampled from this distribution.

The <span style="color:#FF0000">**policy**</span> is denoted as conditional probability $\pi_{\Theta}(a|s)=P(A=a_t|S=s_t)$

with $\sum_a\pi_{\Theta}(a|s)=1 \: \forall s$


> If the policy is our LLM, the action is the output of it, meaning the <span style="color:#FF0000">**next token**</span> $a_t$

> So the LLM is denoted by the probability of generating one token (over the output distribution), given the prompt and the sequence of tokens
> generated in the previous steps 

$$ \pi_{\Theta}(a|s)=P(A=a_t|S=s_t) \rightarrow \pi_{\Theta}(y_t|x, y_{<t})$$

### Environment, state and Transition function

> When the policy outputs an action, <span style="color:red">**the state of the environment will be updated according to
a transition function**</span>, which is part of the environment.

> <span style="color:red">**Transition function**</span>: $P(s_{t+1}|a_t,s_t)$

### Reward

> After an action is taken, the environment provides a <span style="color:red">**reward
**</span> <span style="color:red">**$r_t$**</span> (positive, negative or zero), which characterizes the desirability of
> a given state or action.

Intuitively, we can <span style="color:red">**conceptualize RL as a trial and error process**</span>. As our agent acts
in its environment, the training process reinforces—either positively or negatively—observed behavior via the reward.
Hence, the name “reinforcement” learning.

#### Type of rewards

We distinguish two tipes of reward models

#####  <u>RLHf</u> (Reinforcment learning from Human Feedback)

- used for <span style="color: red;"><strong>subjective</strong></span> problems, it is not possible to objeectively determine the reward; the human feedback is necesary
- the objective function is based on the <span style="color: red;"><strong>Bradley-Terry formulation</strong></span>, based on the difference of rewards of pairs of output sequences.
- the objective function optimizes a <span style="color: red;"><strong>reward model</strong></span>

#####  <u>RLVR</u> (Reinforcment learning with Verifiable Reward)

- the reward can be assigned with deterministic functions (tests for code, or results for calculus, etc)
- The example is reasoning learning


### Process

![rl_training_diagram.png](img/rl_training_diagram.png)

Beginning from an initial state $s_0$ that is sampled from a distribution $d_0$ over the set of initial states,
the agent will repeatedly:

* Sample actions ($a_t$).

* Transition to a new state ($s_{t+1}$).

* Receive a reward ($r_t$).

This process will continue until the final state $s_t$ is reached in case:

* $s_t$ is a <span style="color:#FF0000">**terminal state**</span>
* a pre-defined <span style="color:#FF0000">**maximum number of steps**</span> has been reached

## Policy gradient

### Trajectory / Completion

The sequence of actions, states and rewards

![trajectory.png](img/trajectory.png)

In the LLM world the trajectory is called <span style="color:red">**completion**</span>:

> Given a prompt $x$, a language model generates a <span style="color:red">**completion
**</span> $y=(y_1, y_2, \dots ,y_T)$ one token at a time

The probability of the trajectory, is calculated by the combination of policy and transformation
using the chain rule:

![trajectory_prob.png](img/trajectory_prob.png)

### The optimization process

> Change model weights to <span style="color: red;"><strong>increase rewards</strong></span> to <span style="color: red;"><strong>maximize the expected rewards</strong></span> of the generated outputs of the model/policy

![alt text](img/preference_tuning.png)











> The same way we have the <span style="color:red">**completion probability**</span>:
> $p_{\Theta}(y | x) = \prod_{t=1}^Tp_{\Theta}(y_t|x, y_{\lt t})$

The product of all the probability of the current token geneated at time $t$, given the input and all the
previous tokens ($\lt t$)

Generating a completion traces one path through a tree of possible continuations.

### Return / reward function

For reasoning tasks, the **reward function** is often a verifier $R(x,y)$  if the final answer is correct and

$$
R(x,y)\colon
\left\{
\begin{array}{rcl}
\mathbf{1} & \longrightarrow & \text{correct answer} \\
\mathbf{0} & \longmapsto & \text{otherwise}
\end{array}
\right.
$$

This setup is often called RL with verifiable rewards (RLVR). Math problems with a known answer and coding tasks with
unit tests are common examples.

![return.png](img/return.png)

### The objective function of RL

#### Expectation as average reward

> Our goal is to find the parameters θ (parameters of LLM) that maximize the expected reward, which we call
> the <span style="color:red">**objective J(θ)**</span>
> $J(\Theta)=\mathbb{E}_{x \sim \mathcal{D},y \sim p_{\Theta}(\cdot| x)}[R(x,y)]$

where

* $x\sim\mathcal{D}$: a prompt $x$ drawn from a dataset (distribution) $\mathcal{D}$.
* $y\sim p_\Theta(\cdot\mid x)$: a response $y$ sampled from the model with parameters $\Theta$, given the prompt. The dot means “the distribution over all $y$”.
* $R(x,y)$: a scalar reward for that prompt/response pair (e.g. from a reward model).
* $J(\Theta)$: the average reward the model earns. Training maximizes it.

$$J(\Theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim p_\Theta(\cdot\mid x)}[R(x,y)]=\sum_x \mathcal{D}(x)\sum_y p_\Theta(y\mid x)\,R(x,y)$$

An expectation is $(probability) × (value)$ summed over outcomes.

$$\mathbb{E}(A)=\sum a \cdot P(a)=a_1 \cdot p(a_1) + a_2 \cdot p(a_2) + \cdots a_k \cdot p(a_k)$$

Here the outcomes are pairs $(x,y)$ with probability $\mathcal{D}(x)\,p_\Theta(y\mid x)$.

#### Fix the prompt

We fix the prompt (it is impossible to estimate all possible prompts given to an LLM) <span style="color:red">**$\mathcal{D}(x)=1$**</span>

$$J(\Theta)=\sum_y p_\Theta(y)\,R(y)$$

In general RL, an agent collects a reward after each of many actions, and the objective is the expected total reward
over an episode, often discounted.
<u>Generating a completion is an episode where each token is an action and the only reward comes at the end</u>, so the
total is just $R(y)$

### The policy gradient (gradient with respect policy parameters)

To improve the model, we want to follow the gradient of the objective, $\nabla_\Theta J(\Theta)$  the direction in
parameter space that most increases the expected reward.

Written out, the objective is a sum over every possible completion:

$$\nabla_{\Theta}(J(\Theta))=\nabla_{\Theta}\sum_y p_{\Theta}(y)R(y)=\sum_y \nabla_{\Theta}p_{\Theta}(y)R(y)$$


if we apply a <span style="color:red">**log derivative trick**</span>: 
* $\nabla log(f)=\frac{\nabla f}{f}$ we can write
* $\nabla (f)= f \cdot \nabla log(f)$
* so $\nabla_{\Theta}p_{\Theta}(y)=p_{\Theta}\nabla_{\Theta} \log{p_{\Theta}}$
* The gradient becomes: $\nabla_{\Theta}(J(\Theta))=\sum_y p_{\Theta}(y)\nabla_{\Theta}\log{ p_{\Theta}(y)}R(y)$ which is

> $$\nabla_{\Theta}(J(\Theta)) = \mathbb{E}_{y \sim p_{\Theta}}[R(y)\nabla_{\Theta}\log{ p_{\Theta}(y)}]$$

The $\nabla_{\Theta}\log{ p_{\Theta}(y)}$ is called  <span style="color:red">**score function**</span>

> So the gradient is the expected value of the product of the return and the score function for each
> completion.

Each term pairs a direction (<u>the score</u>) with a weight (<u>the reward</u>): the score points toward making that rollout’s completion more likely, and the
reward sets how much that direction counts.

![monte_carlo_example.png](img/monte_carlo_example.png)

Because all completions share a total probability of 1, B’s and C’s gains come from elsewhere, here mostly from
completions nobody sampled.

A barely changes, because it shares everything up to “340 +” with B, so reinforcing B also lifts most of A’s path.

The numbers are only illustrative: B and C are pushed up, but how every other completion moves depends on how the
model’s parameters are shared.

> Policy gradient is supervised fine-tuning on your own samples, weighted by reward. With +1/0 rewards, as in the figure
> above, it is literally SFT on the correct completions, so incorrect completions are never pushed down directly; they
> only lose share.

### Monte Carlo policy gradient

The calculation of the expected value requires the average of all possible completions, given the prompt, which is
computationally not feasible.

We rather use the 

<span style="color:#FF0000">**Monte Carlo estimate from N randomly sampled trajectories/completions**</span>


$$J(\Theta) \approx \frac{1}{N}\sum_{i=1}^{N}R(y_{i})\nabla_{\Theta}\log{ p_{\Theta}(y_i)} \text{ where } y_{i} \sim p_{\Theta} $$

> It estimates the gradient by averaging over complete sampled rollouts, using each one’s actual reward rather than a
> learned estimate of how good it was.


### From sequence to tokens

With the Monte Carlo estimate we can write the objective function as

$$J(\Theta) \approx \frac{1}{N}\sum_{i=1}^{N}R(y_{i})\nabla_{\Theta}\log{ p_{\Theta}(y_i)} \text{ where } y_{i} \sim p_{\Theta} $$

* we must consider the token contribution to the completion for each timestamp
* the reward function is still about the completion, is is not declined to the single token
* so we elaborate the score function based on the token


<span style="color:red">**The score function for a single completion $\nabla_{\Theta}\log{ p_{\Theta}(y)}$ brakes down to a sum of per-token
log-probabilities (one term per token)**</span>

$$\nabla_{\Theta}\log{ p_{\Theta}(y_i)}=\sum_{t=1}^T\nabla_{\Theta}\log{ p_{\Theta}(y_t|y_{\lt t})}$$

* each term is the **score of each token**: the direction that most increases the probability of choosing $y_t$, given the
  previous generation and the prompt

For example, the current generation is until `340 +`, the LLM will generate a prob distribution for each token $v$ in
the vocabulary, so that for each we have a corresponding logit 

#### Derivative with respect to the logits of Softmax gradient in Claude

$p_v=p_{\Theta}(v|y_{\lt t})$ and $s_v=\nabla_{\Theta}\log{p_{v}}$
 
$z_v \rightarrow p_v=softmax(z_v) = \frac{e^{z_v}}{\sum_ie^{z_i}}$

So we can rewrite the score function )for each completion that we fix with $y$ and not $y_i$



$$\nabla_{\Theta}\log{ p_{\Theta}(y)}=\sum_{t=1}^T\nabla_{\Theta}\log{p_{\Theta}(y_t|y_{\lt t})}$$

but, for each time $t$

$$\nabla_{\Theta}\log{ p_{\Theta}(y_t|y_{\lt t})}=\nabla_{\Theta}\log{ \frac{e^{z_t}}{\sum_k e^{z_k}}}=\frac{\delta }{\delta e^{z_u}}\log{e^{z_t}} -\frac{\delta }{\delta e^{z_u}} \log {\sum_k e^{z_k}}$$

* The first term becomes $\frac{\delta }{\delta e^{z_u}}e^{z_t}$ which is $1[t=u]$
* The second term $\frac{\delta }{\delta e^{z_u}} \log{\sum_k e^{z_k}}$ for the derivation rule $\frac{\partial}{\partial z_u}\ln S=\frac{1}{S}\cdot\frac{\partial S}{\partial z_u}$ becomes
$\frac{1}{\sum_k e^{z_k}}\frac{\partial}{\partial z_u}\sum_k e^{z_k} \: (\alpha)$
* The term $\frac{\partial}{\partial z_u}\sum_k e^{z_k}$ is not null only for $k=u$ and it is $\frac{\partial}{\partial z_u}e^{z_u}=e^{z_u} \: (\beta)$
* so the last two $(\alpha)(\beta)$ are: $\frac{1}{\sum_k e^{z_k}}e^{z_u}=p_u$ 

So, putting everything together we have the score  $s_t=$ 

> $$s_t=\sum_u(1[t=u]-p_u)\nabla z_u$$

![token_scores.png](img/token_scores.png)

<u>GRAY</u>: the softmax probabilities generated at time step $t$ by the policy for the specific token in the completion
(58 for A and 68 for B)

<u>PURPLE</u>: the own logit gradient calculated via the score 

In the picture we have two completions $A$ and $B$ and we look, at the same time step 
$t$, the vocabulary tokens probabilities. In the completions we take the probability generated by the policy
at that time step $t$ (softmax) 



* A: <span style="color:gray">**$p(58|17, x, 24, =, \cdots)=p_{58}=0.07$**</span> (unlikely token, $R_A=0$) so, since $t=u$ we get <span style="color:purple">**$s_t=1-p_{58}=0.93$**</span>
* B: <span style="color:gray">**$p(68|17, x, 24, =, \cdots)=p_{68}=0.82$**</span> so, since $t=u$ we get <span style="color:purple">**$s_t=1-p_{68}=0.18$**</span>


If we apply the rewards $R\cdot(1-p)$

* A: $R\cdot(1-p_{58})=0\times0.93$
* B: $R\cdot(1-p_{68})=1\times0.18$

So, when the reward is 1, we can calculate how the token distribution and the score in the target function changes the 
scores for all the token in the same time step (the same when 68 was generated).

The score gradient gives a boost of 0.18 to the generated token, penalizing of $-p_u$ the other softmax token values
(this way the total sum will be always 1). In the picture the rest of the $~150k$ tokens share the -0.05 remaining probability














<span style="color:#FF0000">**TBC**</span> with Claude explanation of the image




  





We calculate the derivative with a <span style="color:#FF0000">**trick for logs derivative**</span>: $\delta \log{f}=\delta f /f \rightarrow $ <span style="color:#FF0000">**$\delta f=f \delta \log{f}$**</span>  









We know a LLM (mostly decoder-only) transformer based produces probabilities 











| Rule | Name                           | Expression                                                                                                                                               | Used in    |
|------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| A    | Log-derivative                 | $\displaystyle \frac{d}{dx}\ln f(x)=\frac{f'(x)}{f(x)}\;\Longrightarrow\;\nabla f=f\,\nabla\ln f$                                                        | Steps 1, 4 |
| B    | Log of a quotient              | $\displaystyle \ln\frac{x}{y}=\ln x-\ln y$                                                                                                               | Step 3     |
| C    | Log undoes exp                 | $\displaystyle \ln e^{x}=x$                                                                                                                              | Step 3     |
| D    | Derivative of exp inside a sum | $\displaystyle \frac{d}{dz}e^{z}=e^{z},\qquad \frac{\partial}{\partial z_k}\sum_j e^{z_j}=e^{z_k}$                                                       | Step 4     |
| E    | Multivariable chain rule       | $\displaystyle \frac{\partial\log\pi(a)}{\partial\theta_{m,i}}=\sum_k\frac{\partial\log\pi(a)}{\partial z_k}\,\frac{\partial z_k}{\partial\theta_{m,i}}$ | Step 5     |
| F    | Derivative of a linear layer   | $\displaystyle z_k=\sum_i\theta_{k,i}s_i\;\Longrightarrow\;\frac{\partial z_k}{\partial\theta_{m,i}}=\delta_{km}\,s_i$                                   | Step 5     |