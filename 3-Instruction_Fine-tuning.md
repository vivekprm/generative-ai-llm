We explored example use cases for large language models and discussed the kinds of tasks that were capable of carrying out. Now we'll learn about methods that we can use to improve the performance of an existing model for our specific use case.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a5caf32d-6486-44d1-a53f-198f823ba1e6)

We'll also learn about important metrics that can be used to evaluate the performance of your finetuned LLM and quantify its improvement over the base model we started with.

# Fine Tuning an LLM with Instruction Prompts
Earlier, we saw that some models are capable of identifying instructions contained in a prompt and correctly carrying out zero shot inference.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/47396e8e-85ac-4056-8fba-fdd1629a3e85)

while others, such as smaller LLMs, may fail to carry out the task, like the example shown here.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/67840d82-ffb0-479d-be69-4308037898f2)

We also saw that including one or more examples of what we want the model to do, known as one shot or few shot inference, can be enough to help the model identify the task and generate a good completion. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/51e72501-89aa-454b-9545-de4cb6849caa)

However, this strategy has a couple of drawbacks. 

# Limitations of in-context learning
![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/fd6f3425-a80d-4003-b30d-22dd42c76db1)

- First, for smaller models, it doesn't always work, even when five or six examples are included.
- Second, any examples you include in your prompt take up valuable space in the context window, reducing the amount of room you have to include other useful information.

Luckily, another solution exists, you can take advantage of a process known as fine-tuning to further train a base model.

In-contrast to pre-training, where you train the LLM using vast amounts of unstructured textual data via selfsupervised learning, **fine-tuning is a supervised learning process where you use a data set of labeled examples** to update the weights of the LLM. 

The labeled examples are prompt completion pairs, the fine-tuning process extends the training of the model to improve its ability to generate good completions for a specific task. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/848a46bd-0ce7-4d56-8cd9-ad79f7a133e0)

One strategy, known as **instruction fine tuning**, is particularly good at improving a model's performance on a variety of tasks. Let's take a closer look at how this works, instruction fine-tuning trains the model using examples that demonstrate how it should respond to a specific instruction. Here are a couple of example prompts to demonstrate this idea.

