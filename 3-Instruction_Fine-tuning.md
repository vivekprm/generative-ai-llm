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

One strategy, known as **instruction fine tuning**, is particularly good at improving a model's performance on a variety of tasks. 

Let's take a closer look at how this works, instruction fine-tuning trains the model using examples that demonstrate how it should respond to a specific instruction. Here are a couple of example prompts to demonstrate this idea.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/40164c3a-1a83-4287-af16-123077969ffa)

The instruction in both examples is classify this review, and the desired completion is a text string that starts with sentiment followed by either positive or negative.

The data set you use for training includes many pairs of prompt completion examples for the task you're interested in, each of which includes an instruction. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/2a103919-1469-4a17-ba67-5492304e448c)

For example, if you want to fine tune your model to improve its summarization ability, you'd build up a data set of examples that begin with the instruction summarize, the following text or a similar phrase. And if you are improving the model's translation skills, your examples would include instructions like translate this sentence. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/9588571e-234c-4547-869e-4e5e16fb53db)

These prompt completion examples allow the model to learn to generate responses that follow the given instructions.** Instruction fine-tuning, where all of the model's weights are updated is known as full fine-tuning**. The process results in a new version of the model with updated weights. 

It is important to note that just like pre-training, full fine tuning requires enough memory and compute budget to store and process all the gradients, optimizers and other components that are being updated during training. So you can benefit from the memory optimization and parallel computing strategies that you learned earlier.

**So how do you actually go about instruction, fine-tuning and LLM? **
The first step is to prepare your training data. There are many publicly available datasets that have been used to train earlier generations of language models, although most of them are not formatted as instructions. 

Luckily, developers have assembled prompt template libraries that can be used to take existing datasets, for example, the large data set of Amazon product reviews and turn them into instruction prompt datasets for fine-tuning. Prompt template libraries include many templates for different tasks and different data sets. 

Here are three prompts that are designed to work with the Amazon reviews dataset and that can be used to fine tune models for classification, text generation and text summarization tasks. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a51ba66a-2a1d-41a4-b52e-96757d77e6c7)

You can see that in each case you pass the original review, here called review_body, to the template, where it gets inserted into the text that starts with an instruction like predict the associated rating, generate a star review, or give a short sentence describing the following product review. 

The result is a prompt that now contains both an instruction and the example from the data set. Once you have your instruction data set ready, as with standard supervised learning, you divide the data set into training validation and test splits. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/dc649b12-1b65-4818-a6ae-a25bb2b80f0d)

During fine tuning, you select prompts from your training data set and pass them to the LLM, which then generates completions. Next, you compare the LLM completion with the response specified in the training data. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/0c23b760-550f-44a3-a32f-c58040ffbcf4)

You can see here that the model didn't do a great job, it classified the review as neutral, which is a bit of an understatement. The review is clearly very positive. Remember that the output of an LLM is a probability distribution across tokens. So you can compare the distribution of the completion and that of the training label and use the standard crossentropy function to calculate loss between the two token distributions. And then use the calculated loss to update your model weights in standard backpropagation. 

You'll do this for many batches of prompt completion pairs and over several epochs, update the weights so that the model's performance on the task improves. As in standard supervised learning, you can define separate evaluation steps to measure your LLM performance using the holdout validation data set. This will give you the validation accuracy, and after you've completed your fine tuning, you can perform a final performance evaluation using the holdout test data set. This will give you the test accuracy. 

The fine-tuning process results in a new version of the base model, often called an instruct model that is better at the tasks you are interested in. Fine-tuning with instruction prompts is the most common way to fine-tune LLMs these days. 

# Fine-tuning on a Single Task
While LLMs have become famous for their ability to perform many different language tasks within a single model, your application may only need to perform a single task. In this case, you can fine-tune a pre-trained model to improve performance on only the task that is of interest to you. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/05c9f402-0de5-451f-811f-6710ce0020be)

For example, summarization using a dataset of examples for that task. Interestingly, good results can be achieved with relatively few examples. Often just 500-1,000 examples can result in good performance in contrast to the billions of pieces of texts that the model saw during pre-training. 

However, there is a potential downside to fine-tuning on a single task. The process may lead to a phenomenon called **catastrophic forgetting**. 

## Catastrophic Forgetting
Catastrophic forgetting happens because the full fine-tuning process modifies the weights of the original LLM. While this leads to great performance on the single fine-tuning task, it can degrade performance on other tasks. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/467beacb-36fb-4c57-9905-5c53bf6a7d1f)

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/bf10dac1-9466-4427-9aac-afac7b581e86)

For example, while fine-tuning can improve the ability of a model to perform sentiment analysis on a review and result in a quality completion, the model may forget how to do other tasks. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/9340b628-bcab-4499-a490-a842329da43a)

This model knew how to carry out named entity recognition before fine-tuning correctly identifying Charlie as the name of the cat in the sentence. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/db6ca0d1-ddda-4538-87f6-e97537e62e3a)

But after fine-tuning, the model can no longer carry out this task, confusing both the entity it is supposed to identify and exhibiting behavior related to the new task. 

What options do you have to avoid catastrophic forgetting?
- First of all, it's important to decide whether catastrophic forgetting actually impacts your use case. If all you need is reliable performance on the single task you fine-tuned on, it may not be an issue that the model can't generalize to other tasks.
- If you do want or need the model to maintain its multitask generalized capabilities, **you can perform fine-tuning on multiple tasks at one time**. 

Good multitask fine-tuning may require 50-100,000 examples across many tasks, and so will require more data and compute to train. 

Our second option is to perform **Parameter Efficient Fine-tuning**, or PEFT for short instead of full fine-tuning. PEFT is a set of techniques that preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters. PEFT shows greater robustness to catastrophic forgetting **since most of the pre-trained weights are left unchanged**. 

# Multi-task Instruction fine-tuning
