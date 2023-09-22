# Aligning models with human values
Let's come back to the Generative AI project life cycle. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/de8d34ea-9a69-4a4a-8fed-04313d06e145)

Previously, we looked closely at a technique called fine-tuning. The goal of fine-tuning with instructions, including path methods, is to further train your models so that they better understand human like prompts and generate more human-like responses. 

This can improve a model's performance substantially over the original pre-trained based version, and lead to more natural sounding language. 

However, natural sounding human language brings a new set of challenges. By now, you've probably seen plenty of headlines about large language models behaving badly. Issues include models using:
- Toxic language in their completions.
- Replying in combative and aggressive voices.
- Providing detailed information about dangerous topics.

These problems exist because large models are trained on vast amounts of texts data from the Internet where such language appears frequently. 
Here are some examples of models behaving badly. 

Let's assume you want your LLM to tell you knock, knock, joke, and the models responses just clap, clap. While funny in its own way, it's not really what you were looking for. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/2236b765-627c-4f15-b1fa-dd2baac638f4)

The completion here is not a helpful answer for the given task. 

Similarly, the LLM might give misleading or simply incorrect answers. If you ask the LLM about the disproven Ps of health advice like coughing to stop a heart attack, the model should refute this story. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/4516ffec-2e27-4e7c-a4e0-74d79d9c41d7)

Instead, the model might give a confident and totally incorrect response, definitely not the truthful and honest answer a person is seeking. 

Also, the LLM shouldn't create harmful completions, such as being offensive, discriminatory, or eliciting criminal behavior, as shown here, when you ask the model how to hack your neighbor's WiFi and it answers with a valid strategy. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/36617977-a44e-4bf1-95de-8f1f8f6b15fc)

Ideally, it would provide an answer that does not lead to harm.

These important human values, **helpfulness, honesty, and harmlessness** are sometimes collectively called HHH, and are a set of principles that guide developers in the responsible use of AI. 
**Additional fine-tuning with human feedback helps to better align models with human preferences and to increase the helpfulness, honesty, and harmlessness of the completions**. This further training can also help to decrease the toxicity, often models responses and reduce the generation of incorrect information. 

# Reinforcement learning from human feedback (RLHF)
Let's consider the task of text summarization, where you use the model to generate a short piece of text that captures the most important points in a longer article. Your goal is to use fine-tuning to improve the model's ability to summarize, by showing it examples of human generated summaries. 

In 2020, researchers at OpenAI published a paper that explored the use of fine-tuning with human feedback to train a model to write short summaries of text articles. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/002d72a4-abe3-413e-8d29-186859d0a937)

Here you can see that a model fine-tuned on human feedback produced better responses than a pretrained model, an instruct fine-tuned model, and even the reference human baseline. A popular technique to finetune large language models with human feedback is called **reinforcement learning from human feedback**, or RLHF for short.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/6e7a990e-a751-4726-849a-bb4fa055196a)

RLHF uses reinforcement learning, or RL for short, to finetune the LLM with human feedback data, resulting in a model that is better aligned with human preferences. You can use RLHF to make sure that your model produces outputs that maximize usefulness and relevance to the input prompt. 

Perhaps most importantly, RLHF can help minimize the potential for harm. You can train your model to give caveats that acknowledge their limitations and to avoid toxic language and topics.

One potentially exciting application of RLHF is the personalizations of LLMs, where models learn the preferences of each individual user through a continuous feedback process. This could lead to exciting new technologies like individualized learning plans or personalized AI assistants. 

But in order to understand how these future applications might be made possible, let's start by taking a closer look at how RLHF works. 

In case you aren't familiar with reinforcement learning, here's a high level overview of the most important concepts. **Reinforcement learning is a type of machine learning in which an agent learns to make decisions related to a specific goal by taking actions in an environment, with the objective of maximizing some notion of a cumulative reward**.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/48257e68-ebbb-462d-888f-ae8ea92aedc6)

In this framework, the agent continually learns from its experiences by taking actions, observing the resulting changes in the environment, and receiving rewards or penalties, based on the outcomes of its actions. By iterating through this process, the agent gradually refines its strategy or policy to make better decisions and increase its chances of success.

A useful example to illustrate these ideas is training a model to play Tic-Tac-Toe. Let's take a look. In this example, the agent is a model or policy acting as a Tic-Tac-Toe player. Its objective is to win the game. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/85696b5f-eae0-4aa1-9d09-f158449c9dce)

The environment is the three by three game board, and the state at any moment, is the current configuration of the board. The action space comprises all the possible positions a player can choose based on the current board state. The agent makes decisions by following a strategy known as the RL policy. Now, as the agent takes actions, it collects rewards based on the actions' effectiveness in progressing towards a win. 

The goal of reinforcement learning is for the agent to learn the optimal policy for a given environment that maximizes their rewards. This learning process is iterative and involves trial and error. Initially, the agent takes a random action which leads to a new state. From this state, the agent proceeds to explore subsequent states through further actions. The series of actions and corresponding states form a playout, often called a **rollout**. As the agent accumulates experience, it gradually uncovers actions that yield the highest long-term rewards, ultimately leading to success in the game. 

Now let's take a look at how the Tic-Tac-Toe example can be extended to the case of fine-tuning large language models with RLHF. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/e23c9699-84d7-4a8e-9c8e-a86d7d767c28)

In this case, the agent's policy that guides the actions is the LLM, and its objective is to generate text that is perceived as being aligned with the human preferences. This could mean that the text is, for example, helpful, accurate, and non-toxic. The environment is the context window of the model, the space in which text can be entered via a prompt. The state that the model considers before taking an action is the current context. That means any text currently contained in the context window. The action here is the act of generating text. This could be a single word, a sentence, or a longer form text, depending on the task specified by the user. The action space is the token vocabulary, meaning all the possible tokens that the model can choose from to generate the completion. 

How an LLM decides to generate the next token in a sequence, depends on the **statistical representation of language that it learned during its training**. At any given moment, the action that the model will take, meaning which token it will choose next, depends on the prompt text in the context and the probability distribution over the vocabulary space. **The reward is assigned based on how closely the completions align with human preferences.** Given the variation in human responses to language, determining the reward is more complicated than in the Tic-Tac-Toe example.

One way you can do this is to have a human evaluate all of the completions of the model against some alignment metric, such as determining whether the generated text is toxic or non-toxic. This feedback can be represented as a scalar value, either a zero or a one. 

The LLM weights are then updated iteratively to maximize the reward obtained from the human classifier, enabling the model to generate non-toxic completions. However, obtaining human feedback can be time consuming and expensive. As a practical and scalable alternative, you can use an additional model, known as the reward model, to classify the outputs of the LLM and evaluate the degree of alignment with human preferences. 

You'll start with a smaller number of human examples to train the secondary model by your traditional supervised learning methods. Once trained, you'll use the reward model to assess the output of the LLM and assign a reward value, which in turn gets used to update the weights off the LLM and train a new human aligned version. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/aa539ac8-c953-43e4-9393-0691054060fd)

Exactly how the weights get updated as the model completions are assessed, depends on the algorithm used to optimize the policy. Lastly, note that **in the context of language modeling, the sequence of actions and states is called a rollout**, instead of the term playout that's used in classic reinforcement learning. 

The reward model is the central component of the reinforcement learning process. It encodes all of the preferences that have been learned from human feedback, and it plays a central role in how the model updates its weights over many iterations.

# RLHF: Obtaining feedback from humans
## Prepare dataset for Human Feedback
The first step in fine-tuning an LLM with RLHF is to select a model to work with and use it to prepare a data set for human feedback. The model you choose should have some capability to carry out the task you are interested in, whether this is text summarization, question answering or something else. In general, you may find it easier to start with an instruct model that has already been fine tuned across many tasks and has some general capabilities. You'll then use this LLM along with a prompt data set to generate a number of different responses for each prompt. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/c64ec21a-1137-4f24-a48a-2363d8295da4)

The prompt dataset is comprised of multiple prompts, each of which gets processed by the LLM to produce a set of completions. 

The next step is to collect feedback from human labelers on the completions generated by the LLM. This is the human feedback portion of reinforcement learning with human feedback. 

## Collect Human Feedback
- First, you must decide what criterion you want the humans to assess the completions on. This could be any of the issues discussed so far like helpfulness or toxicity.
- Once you've decided, you will then ask the labelers to assess each completion in the data set based on that criterion. Let's take a look at an example.

In this case, the prompt is, my house is too hot. You pass this prompt to the LLM, which then generates three different completions. The task for your labelers is to rank the three completions in order of helpfulness from the most helpful to least helpful. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/2d869512-6ca4-45ab-b1fa-dbc9c785669a)

So here the labeler will probably decide that completion two is the most helpful. It tells the user something that can actually cool their house and ranks as completion first. Neither completion one or three are very helpful, but maybe the labeler will decide that three is the worst of the two because the model actively disagrees with the input from the user. 

So the labeler ranks the top completion second and the last completion third. This process then gets repeated for many prompt completion sets, building up a data set that can be used to train the reward model that will ultimately carry out this work instead of the humans. 

The same prompt completion sets are **usually assigned to multiple human labelers to establish consensus and minimize the impact of poor labelers in the group**. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/6a0294f7-fc0e-42cf-9510-16ec4dd1fdc7)

Like the third labeler here, whose responses disagree with the others and may indicate that they misunderstood the instructions, this is actually a very important point. The clarity of your instructions can make a big difference on the quality of the human feedback you obtain. **Labelers are often drawn from samples of the population that represent diverse and global thinking**.

### Sample Instructions For Human Labelers
Here you can see an example set of instructions written for human labelers. This would be presented to the labeler to read before beginning the task and made available to refer back to as they work through the dataset. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/8a97412b-978c-4470-9b16-48fc29fc0494)

The instructions start with the overall task the labeler should carry out. In this case, to choose the best completion for the prompt. The instructions continue with additional details to guide the labeler on how to complete the task. In general, the more detailed you make these instructions, the higher the likelihood that the labelers will understand the task they have to carry out and complete it exactly as you wish. 

For instance, in the second instruction item, the labelers are told that they should make decisions based on their perception of the correctness and informativeness of the response. They are told they can use the Internet to fact check and find other information.

They are also given clear instructions about what to do if they identify a tie, meaning a pair of completions that they think are equally correct and informative. The labelers are told that it is okay to rank two completions the same, but they should do this sparingly.

A final instruction worth calling out here is what to do in the case of a nonsensical confusing or irrelevant answer. In this case, labelers should select F rather than rank, so the poor quality answers can be easily removed. Providing a detailed set of instructions like this increases the likelihood that the responses will be high quality and that individual humans will carry out the task in a similar way to each other. This can help ensure that the ensemble of labeled completions will be representative of a consensus point of view.

Once your human labelers have completed their assessments off the Prom completion sets, you have all the data you need to train the reward model. Which you will use instead of humans to classify model completions during the reinforcement learning finetuning process. Before you start to train the reward model, however, you need to convert the ranking data into a pairwise comparison of completions. 

In other words, all possible pairs of completions from the available choices to a prompt should be classified as 0 or 1 score. 

In the example shown here, there are three completions to a prompt, and the ranking assigned by the human labelers was 2, 1, 3, as shown, where 1 is the highest rank corresponding to the most preferred response. With the three different completions, there are three possible pairs purple-yellow, purple-green and yellow-green. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a01ec647-a67d-4fde-a14a-8e16250bbc37)

Depending on the number N of alternative completions per prompt, you will have N choose two combinations. For each pair, you will assign a reward of 1 for the preferred response and a reward of 0 for the less preferred response. Then you'll reorder the prompts so that the preferred option comes first. This is an important step because the reward model expects the preferred completion, which is referred to as Yj first. 

Once you have completed this data, restructuring, the human responses will be in the correct format for training the reward model. Note that while thumbs-up, thumbs-down feedback is often easier to gather than ranking feedback, ranked feedback gives you more prom completion data to train your reward model. As you can see, here you get three prompt completion pairs from each human ranking.

## RLHF: Reward model
At this stage, we have everything we need to train the reward model. While it has taken a fair amount of human effort to get to this point, by the time we're done training the reward model, we won't need to include any more humans in the loop. Instead, the reward model will effectively take place off the human labeler and automatically choose the preferred completion during the RLHF process. 

This reward model is usually also a language model. For example, **Bird** that is trained using supervised learning methods on the pairwise comparison data that you prepared from the human labelers assessment off the prompts. For a given prompt X, the reward model learns to favor the human-preferred completion y_ j, while minimizing the lock sigmoid off the reward difference, r_j-r_k.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/f478fb34-8fa1-426b-b63b-28c8df69cd9e)

The human-preferred option is always the first one **labeled y_j**. Once the model has been trained on the human rank prompt-completion pairs, you can use the reward model as a binary classifier to provide a set of logics across the positive and negative classes. Logics are the unnormalized model outputs before applying any activation function. Let's say you want to detoxify your LLM, and the reward model needs to identify if the completion contains hate speech. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/95df03b4-d365-4b28-9bfe-167af425f639)

In this case, the two classes would be notate, the positive class that you ultimately want to optimize for and hate the negative class you want to avoid. The largest value of the positive class is what you use as the reward value in LLHF. 

Just to remind you, if you apply a Softmax function to the logits, you will get the probabilities. The example here shows a good reward for non-toxic completion and the second example shows a bad reward being given for toxic completion. 

# RLHF: Fine-tuning with reinforcement learning
Let's bring everything together, and look at how we will use the reward model in the reinforcement learning process to update the LLM weights, and produce a human aligned model.

Remember, you want to start with a model that already has good performance on your task of interests. You'll work to align an instruction fine tuned LLM. 
First, you'll pass a prompt from your prompt dataset. In this case, a dog is, to the instruct LLM, which then generates a completion, in this case a furry animal. 

Next, you sent this completion, and the original prompt to the reward model as the prompt completion pair. The reward model evaluates the pair based on the human feedback it was trained on, and returns a reward value. A higher value such as 0.24 as shown here represents a more aligned response. A less aligned response would receive a lower value, such as negative 0.53. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/6803d5db-6338-476c-a922-62ea0cbba19c)

You'll then pass this reward value for the prompt completion pair to the reinforcement learning algorithm to update the weights of the LLM, and move it towards generating more aligned, higher reward responses. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/21c3c3ff-694d-4397-ab35-b12d64e5a5e4)

Let's call this intermediate version of the model the RL updated LLM. These series of steps together forms a single iteration of the RLHF process. These iterations continue for a given number of epics, similar to other types of fine tuning. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/1110b7d5-ab67-448e-8770-5dee6ca1ff4b)

Here you can see that the completion generated by the RL updated LLM receives a higher reward score, indicating that the updates to weights have resulted in a more aligned completion. If the process is working well, you'll see the reward improving after each iteration as the model produces text that is increasingly aligned with human preferences. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/3684c518-2c83-4819-b4ab-27a0661907dc)

You will continue this iterative process until your model is aligned based on some evaluation criteria. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a6eca77e-4eba-4adf-a316-807b431653e8)

For example, reaching a threshold value for the helpfulness you defined. 

You can also define a maximum number of steps, for example, 20,000 as the stopping criteria. At this point, let's refer to the fine-tuned model as the **human-aligned LLM**. 

One detail we haven't discussed yet is the exact nature of the reinforcement learning algorithm. This is the algorithm that takes the output of the reward model and uses it to update the LLM model weights so that the reward score increases over time. There are several different algorithms that you can use for this part of the RLHF process. A popular choice is **proximal policy optimization** or PPO for short. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/9c2b1235-07a1-4d18-960a-9c7f7b5b0142)

PPO is a pretty complicated algorithm, and you don't have to be familiar with all of the details to be able to make use of it. However, it can be a tricky algorithm to implement and understanding its inner workings in more detail can help you troubleshoot if you're having problems getting it to work.

# Proximal Policy Optimization
PPO stands for Proximal Policy Optimization, which is a powerful algorithm for solving reinforcement learning problems. As the name suggests, PPO optimizes a policy, in this case the LLM, to be more aligned with human preferences. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/abcafddb-d284-4298-ba41-0cfee5ee692f)

Over many iterations, PPO makes updates to the LLM. The updates are small and within a bounded region, resulting in an updated LLM that is close to the previous version, hence the name Proximal Policy Optimization. 

Keeping the changes within this small region result in a more stable learning. The goal is to update the policy so that the reward is maximized. How?

You start PPO with your initial instruct LLM, then at a high level, each cycle of PPO goes over two phases. 
In Phase I, the LLM, is used to carry out a number of experiments, completing the given prompts. These experiments allow you to update the LLM against the reward model in Phase II. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/cdcc1757-01b7-415c-8adb-577257e40590)

Remember that the reward model captures the human preferences. For example, the reward can define how helpful, harmless, and honest the responses are. The expected reward of a completion is an important quantity used in the PPO objective. We estimate this quantity through a separate head of the LLM called the **value function**. Let's have a closer look at the value function and the value loss.

## Calculate Rewards
Assume a number of prompts are given. First, you generate the LLM responses to the prompts, then you calculate the reward for the prompt completions using the reward model. 

For example, the first prompt completion shown here might receive a reward of 1.87. The next one might receive a reward of -1.24, and so on. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/8ab3daef-62f4-4e71-adea-fe34c6f7ab4c)

You have a set of prompt completions and their corresponding rewards. 

## Calculate Value Loss
The value function estimates the expected total reward for a given State S. In other words, as the LLM generates each token of a completion, you want to estimate the total future reward based on the current sequence of tokens. You can think of this as a baseline to evaluate the quality of completions against your alignment criteria. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/456d7102-26ed-4777-900c-ed728037c20d)

Let's say that at this step of completion, the estimated future total reward is 0.34. 

With the next generated token, the estimated future total reward increases to 1.23. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/f385a940-4db9-442a-a48c-6649f149d70a)

The goal is to minimize the value loss that is the difference between the actual future total reward in this example, 1.87, and its approximation to the value function, in this example, 1.23. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/7b46cdb6-0723-4592-93f9-b15f88db4f5f)

The value loss makes estimates for future rewards more accurate. The value function is then used in Advantage Estimation in Phase 2, which we will discuss in a bit.
This is similar to when you start writing a passage, and you have a rough idea of its final form even before you write it. 

You mentioned that the losses and rewards determined in Phase 1 are used in Phase 2 to update the weights resulting in an updated LLM. Can you explain this process in a little bit more detail? 
Sure. In Phase 2, you make a small updates to the model and evaluate the impact of those updates on your alignment goal for the model. The model weights updates are guided by the prompt completion, losses, and rewards. PPO also ensures to keep the model updates within a certain small region called the **trust region**. This is where the proximal aspect of PPO comes into play. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/b40ab6eb-27ce-4ab6-88e2-870246504366)

Ideally, this series of small updates will move the model towards higher rewards. The PPO policy objective is the main ingredient of this method.

Remember, the objective is to find a policy whose expected reward is high. In other words, you're trying to make updates to the LLM weights that result in completions more aligned with human preferences and so receive a higher reward. 

The policy loss is the main objective that the PPO algorithm tries to optimize during training. I know the math looks complicated, but it's actually simpler than it appears. Let's break it down step-by-step. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/c102f512-dc78-488d-a895-0586099e99eb)

First, focus on the most important expression and ignore the rest for now. Pi of A_t given S_t in this context of an LLM, is the probability of the next token A_t given the current prompt S_t. The action A_t is the next token, and the state S_t is the completed prompt up to the token t. 
The denominator is the probability of the next token with the initial version of the LLM which is frozen. The numerator is the probabilities of the next token, through the updated LLM, which we can change for the better reward. 

A^<sub>t</sub> is called the estimated advantage term of a given choice of action. The advantage term estimates how much better or worse the current action is compared to all possible actions at data state. We look at the expected future rewards of a completion following the new token, and we estimate how advantageous this completion is compared to the rest.

There is a recursive formula to estimate this quantity based on the value function that we discussed earlier. Here, we focus on intuitive understanding. Here is a visual representation of what I just described. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/be86bd8e-333f-4891-85d3-e20d70d8db54)

You have a prompt S, and you have different paths to complete it, illustrated by different paths on the figure. The advantage term tells you how better or worse the current token a<sub>t</sub> is with respect to all the possible tokens. In this visualization, the top path which goes higher is better completion, receiving a higher reward. The bottom path goes down which is a worst completion. 

Why does maximizing this term lead to higher rewards? 
Let's consider the case where the advantage is positive for the suggested token. A positive advantage means that the suggested token is better than the average. Therefore, increasing the probability of the current token seems like a good strategy that leads to higher rewards. This translates to maximizing the expression we have here. 
If the suggested token is worse than average, the advantage will be negative. Again, maximizing the expression will demote the token, which is the correct strategy. 

So the overall conclusion is that maximizing this expression results in a better aligned LLM. Great. So let's just maximize this expression then?
Directly maximizing the expression would lead into problems because our calculations are reliable under the assumption that our advantage estimations are valid. 
The advantage estimates are valid **only when the old and new policies are close to each other**. This is where the rest of the terms come into play. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/9022c8e1-abe9-41c3-9037-0c3140250c86)

So stepping back and looking at the whole equation again, what happens here is that you pick the smaller of the two terms. The one we just discussed and this second modified version of it. Notice that this second expression defines a region, where two policies are near each other. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/fc91a4ba-d015-4da5-86bb-a255d3c2431a)

These extra terms are guardrails, and simply define a region in proximity to the LLM, where our estimates have small errors. This is called the trust region. 
These extra terms ensure that we are unlikely to leave the trust region. In summary, optimizing the PPO policy objective results in a better LLM without overshooting to unreliable regions. 

Are there any additional components? 
Yes. You also have the entropy loss. While the policy loss moves the model towards alignment goal, entropy allows the model to maintain creativity. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/e99592f9-a268-4ba0-86b2-81e684232fbc)

If you kept entropy low, you might end up always completing the prompt in the same way as shown here. Higher entropy guides the LLM towards more creativity.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/87bb90cc-61e5-4a4b-bf47-12aaead87b6c)

This is similar to the temperature setting of LLM that you've seen earlier. The difference is that the **temperature influences model creativity at the inference time**, while the **entropy influences the model creativity during training**. 

Putting all terms together as a weighted sum, we get our PPO objective, which updates the model towards human preference in a stable manner. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/f7aec75c-f592-4a23-bcf7-5ee95de2733a)

This is the overall PPO objective. The C1 and C2 coefficients are hyperparameters. The PPO objective updates the model weights through back propagation over several steps. 

Once the model weights are updated, PPO starts a new cycle. For the next iteration, the LLM is replaced with the updated LLM, and a new PPO cycle starts. 

After many iterations, you arrive at the human-aligned LLM. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/2674caf3-ac67-4246-8b48-3222dc773a86)

Now, are there other reinforcement learning techniques that are used for RLHF?
Q-learning is an alternate technique for fine-tuning LLMs through RL, but PPO is currently the most popular method. In my opinion, PPO is popular because it has the right balance of complexity and performance. That being said, **fine-tuning the LLMs through human or AI feedback is an active area of research**. We can expect many more developments in this area in the near future. 

For example, just before we were recording this video, researchers at Stanford published a paper describing a technique called **direct preference optimization**, which is a simpler alternate to RLHF. 

# RLHF: Reward hacking
RLHF is a fine-tuning process that aligns LLMs with human preferences. In this process, you make use of a reward model to assess and LLMs completions of a prompt data set against some human preference metric, like helpful or not helpful. 

Next, you use a reinforcement learning algorithm, in this case, PPO, to update the weights off the LLM based on the reward is signed to the completions generated by the current version off the LLM.

You'll carry out this cycle of a multiple iterations using many different prompts and updates off the model weights until you obtain your desired degree of alignment. Your end result is a **human aligned LLM** that you can use in your application. 

An interesting problem that can emerge in reinforcement learning is known as **reward hacking**, where the agent learns to cheat the system by favoring actions that maximize the reward received even if those actions don't align well with the original objective. In the context of LLMs, reward hacking can manifest as the addition of words or phrases to completions that result in high scores for the metric being aligned. But that reduce the overall quality of the language. 

For example, suppose you are using RLHF to detoxify and instruct model. You have already trained a reward model that can carry out sentiment analysis and classify model completions as toxic or non-toxic. You select a prompt from the training data this product is, and pass it to the instruct an LLM which generates a completion.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/73503db7-b6e2-4566-b697-1473807e609e)

This one, "complete garbage" is not very nice and you can expect it to get a high toxic rating. The completion is processed by the toxicity of reward model, which generates a score and this is fed to the PPO algorithm, which uses it to update the model weights. 

As you iterate RLHF will update the LLM to create a less toxic responses. However, as the policy tries to optimize the reward, it can diverge too much from the initial language model. In this example, the model has started generating completions that it has learned will lead to very low toxicity scores by including phrases like most awesome, most incredible.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/483242ac-b287-483d-92cc-c4eab1b21c0d)

This language sounds very exaggerated. The model could also start generating nonsensical, grammatically incorrect text that just happens to maximize the rewards in a similar way, outputs like this are definitely not very useful. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a06678fe-7c90-4906-9a01-77a3f61fde47)

## Avoiding Reward Hacking 
To prevent reward hacking from happening, you can use the initial instruct LLM as performance reference. Let's call it the reference model. The weights of the reference model are frozen and are not updated during iterations of RHF. This way, you always maintain a single reference model to compare to. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/87b4a8ee-016a-4f06-91f0-63fab0155c0f)

During training, each prompt is passed to both models, generating a completion by the reference LLM and the intermediate LLM updated model. At this point, you can compare the two completions and calculate a value called the Kullback-Leibler divergence, or KL divergence for short. KL divergence is a statistical measure of how different two probability distributions are. You can use it to compare the completions off the two models and determine how much the updated model has diverged from the reference. 

Don't worry too much about the details of how this works. The KL divergence algorithm is included in many standard machine learning libraries and you can use it without knowing all the math behind it. 

KL divergence is calculated for each generate a token across the whole vocabulary off the LLM. This can easily be tens or hundreds of thousands of tokens. However, using a softmax function, you've reduced the number of probabilities to much less than the full vocabulary size. Keep in mind that this is still a relatively compute expensive process. You will almost always benefit from using GPUs.

Once you've calculated the KL divergence between the two models, you added a term to the reward calculation. This will penalize the RL updated model if it shifts too far from the reference LLM and generates completions that are too different. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/585e5bcb-1226-446b-960b-ec6598c842c9)

Note that **you now need to full copies of the LLM to calculate the KL divergence, the frozen reference LLM, and the oral updated PPO LLM**. By the way, you can benefit from combining our relationship with PEFT. In this case, you only update the weights of a path adapter, not the full weights of the LLM. 

This means that you can **reuse the same underlying LLM** for both the reference model and the PPO model, which you update with a trained path parameters. This reduces the memory footprint during training by approximately half. 

I know that there is a lot to take in here, but don't worry, RLHF with PEFT is going to be covered in the lab. If you'll get a chance to see this in action and try it out for yourself. Once you have completed your RLHF alignment of the model, you will want to assess the model's performance. You can use the summarization data set to quantify the reduction in toxicity, for example, the dialogue, some data set that you saw earlier in the course. 

The number you'll use here is the toxicity score, this is the probability of the negative class, in this case, a toxic or hateful response averaged across the completions. If RHF has successfully reduce the toxicity of your LLM, this score should go down. First, you'll create a baseline toxicity score for the original instruct LLM by evaluating its completions off the summarization data set with a reward model that can assess toxic language. Then you'll evaluate your newly human aligned model on the same data set and compare the scores. In this example, the toxicity score has indeed decreased after Arlo HF, indicating a less toxic, better aligned model.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a458988b-b9a1-465d-a0a7-1c042519768e)

# KL Divergence
![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/f50e9598-6cf0-41a4-aa1d-25b4b5757952)

KL-Divergence, or Kullback-Leibler Divergence, is a concept often encountered in the field of reinforcement learning, particularly when using the Proximal Policy Optimization (PPO) algorithm. **It is a mathematical measure of the difference between two probability distributions**, which helps us understand how one distribution differs from another. In the context of PPO, KL-Divergence plays a crucial role in guiding the optimization process to ensure that the updated policy does not deviate too much from the original policy.

In PPO, the goal is to find an improved policy for an agent by iteratively updating its parameters based on the rewards received from interacting with the environment. However, updating the policy too aggressively can lead to unstable learning or drastic policy changes. To address this, **PPO introduces a constraint that limits the extent of policy updates. This constraint is enforced by using KL-Divergence**.

To understand how KL-Divergence works, imagine we have two probability distributions: the distribution of the original LLM, and a new proposed distribution of an RL-updated LLM. KL-Divergence measures the average amount of information gained when we use the original policy to encode samples from the new proposed policy. By minimizing the KL-Divergence between the two distributions, PPO ensures that the updated policy stays close to the original policy, preventing drastic changes that may negatively impact the learning process.

A library that you can use to train transformer language models with reinforcement learning, using techniques such as PPO, is TRL (Transformer Reinforcement Learning). In [this link](https://huggingface.co/blog/trl-peft) you can read more about this library, and its integration with PEFT (Parameter-Efficient Fine-Tuning) methods, such as LoRA (Low-Rank Adaption). The image shows an overview of the PPO training setup in TRL.

# Scaling Human Feedback
Although you can use a reward model to eliminate the need for human evaluation during RLHF fine tuning, **the human effort required to produce the trained reward model in the first place is huge**. The labeled data set used to train the reward model typically requires large teams of labelers, sometimes many thousands of people to evaluate many prompts each. 

This work requires a lot of time and other resources which can be important limiting factors. As the number of models and use cases increases, human effort becomes a limited resource.

Methods to scale human feedback are an active area of research. One idea to overcome these limitations is to scale through model self supervision. Constitutional AI is one approach of scale supervision. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/4922bf56-31ec-4d22-850a-eebf4e031322)

First proposed in 2022 by researchers at Anthropic, Constitutional AI is a method for training models using a set of rules and principles that govern the model's behavior. Together with a set of sample prompts, these form the constitution. You then train the model to self critique and revise its responses to comply with those principles. Constitutional AI is useful not only for scaling feedback, it can also help address some unintended consequences of RLHF. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/0e4633a6-6a78-4881-84a8-172d95edbc9d)

For example, depending on how the prompt is structured, an aligned model may end up revealing harmful information as it tries to provide the most helpful response it can. As an example, imagine you ask the model to give you instructions on how to hack your neighbor's WiFi.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/240638b4-dfd9-492d-8366-22327e9708f0)

Because this model has been aligned to prioritize helpfulness, it actually tells you about an app that lets you do this, even though this activity is illegal. Providing the model with a set of constitutional principles can help the model balance these competing interests and minimize the harm. Here are some example rules from the research paper that Constitutional AI I asks LLMs to follow. 

## Example of Constitutional Principles
For example, you can tell the model to choose the response that is the most helpful, honest, and harmless. But you can play some bounds on this, asking the model to prioritize harmlessness by assessing whether it's response encourages illegal, unethical, or immoral activity. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a59f0eb1-9c32-42fc-93b1-9d802c32dff3)

Note that you don't have to use the rules from the paper, you can define your own set of rules that is best suited for your domain and use case. When implementing the Constitutional AI method, you train your model in two distinct phases. In the first stage, you carry out supervised learning, to start your prompt the model in ways that try to get it to generate harmful responses, this process is called red teaming. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/373f23cf-1ef2-4638-b1a2-1ff1f2e969b8)

You then ask the model to critique its own harmful responses according to the constitutional principles and revise them to comply with those rules. Once done, you'll fine-tune the model using the pairs of red team prompts and the revised constitutional responses. Let's look at an example of how one of these prompt completion pairs is generated. 

Let's return to the WiFi hacking problem. As you saw earlier, this model gives you a harmful response as it tries to maximize its helpfulness.

To mitigate this, you augment the prompt using the harmful completion and a set of predefined instructions that ask the model to critique its response. Using the rules outlined in the Constitution, the model detects the problems in its response. In this case, it correctly acknowledges that hacking into someone's WiFi is illegal. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/19408457-a214-4887-ae7f-272133eefffe)

Lastly, you put all the parts together and ask the model to write a new response that removes all of the harmful or illegal content. The model generates a new answer that puts the constitutional principles into practice and does not include the reference to the illegal app. 

The original red team prompt, and this final constitutional response can then be used as training data. You'll build up a data set of many examples like this to create a fine-tuned NLM that has learned how to generate constitutional responses. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/57a98c9e-59f3-4bca-baf9-e2a36ea69503)

The second part of the process performs reinforcement learning. This stage is similar to RLHF, except that instead of human feedback, **we now use feedback generated by a model**. This is sometimes referred to as **reinforcement learning from AI feedback or RLAIF**. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/fb09c299-065b-48e0-93d9-57726ce4d94c)

Here you use the fine-tuned model from the previous step to generate a set of responses to your prompt. You then ask the model which of the responses is preferred according to the constitutional principles. The result is a model generated preference dataset that you can use to train a reward model. With this reward model, you can now fine-tune your model further using a reinforcement learning algorithm like PPO, as discussed earlier. 

Aligning models is a very important topic and an active area of research. The foundations of RLHF that you've explored in this lesson will allow you to follow along as the field evolves. I'm really excited to see what new discoveries researchers make in this area.

# Lab 3
```sh
aws s3 cp --recursive s3://dlai-generative-ai/labs/w3-233794/ ./
```
