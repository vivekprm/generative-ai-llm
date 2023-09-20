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

One potentially exciting application of RLHF is the personalizations of LLMs, where models learn the preferences of each individual user through a continuous feedback process. This could lead to exciting new technologies like individualized learning plans or personalized AI assistants. But in order to understand how these future applications might be made possible, let's start by taking a closer look at how RLHF works. In case you aren't familiar with reinforcement learning, here's a high level overview of the most important concepts. Reinforcement learning is a type of machine learning in which an agent learns to make decisions related to a specific goal by taking actions in an environment, with the objective of maximizing some notion of a cumulative reward.
