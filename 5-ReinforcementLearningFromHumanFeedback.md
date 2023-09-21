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

![Uploading image.png…]()

Like the third labeler here, whose responses disagree with the others and may indicate that they misunderstood the instructions, this is actually a very important point. The clarity of your instructions can make a big difference on the quality of the human feedback you obtain. **Labelers are often drawn from samples of the population that represent diverse and global thinking**.
