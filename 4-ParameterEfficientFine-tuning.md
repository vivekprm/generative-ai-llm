Training LLMs is computationally intensive. Full fine-tuning requires memory not just to store the model, but various other parameters that are required during the training process. Even if your computer can hold the model weights, which are now on the order of hundreds of gigabytes for the largest models, you must also be able to allocate memory for optimizer states, gradients, forward activations, and temporary memory throughout the training process. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/42a0e076-7206-4d3e-a47f-5f95e82dc66e)

These additional components can be many times larger than the model and can quickly become too large to handle on consumer hardware. 

# Parameter Efficient Fine Tuning
In contrast to full fine-tuning where every model weight is updated during supervised learning, parameter efficient fine tuning methods **only update a small subset of parameters**. Some path techniques freeze most of the model weights and focus on fine tuning a subset of existing model parameters, for example, particular layers or components. 

Other techniques don't touch the original model weights at all, and instead add a small number of new parameters or layers and fine-tune only the new components. With PEFT, most if not all of the LLM weights are kept frozen. As a result, the number of trained parameters is much smaller than the number of parameters in the original LLM. In some cases, just 15-20% of the original LLM weights. This makes the memory requirements for training much more manageable. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/eb26cb66-79a8-45f2-822e-317c2477a54a)

In fact, **PEFT can often be performed on a single GPU**. And because the original LLM is only slightly modified or left unchanged, PEFT is **less prone to the catastrophic forgetting problems** of full fine-tuning. 

Full fine-tuning results in a new version of the model for every task you train on. Each of these is the same size as the original model, so it can create an expensive storage problem if you're fine-tuning for multiple tasks. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/c82660c0-5ca7-4d94-81a8-40f0c643e674)

Let's see how you can use PEFT to improve the situation. With parameter efficient fine-tuning, you train only a small number of weights, which results in a much smaller footprint overall, as small as megabytes depending on the task. **The new parameters are combined with the original LLM weights for inference**. 

The PEFT weights are trained for each task and can be easily swapped out for inference, allowing efficient adaptation of the original model to multiple tasks.  

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/e7385beb-8104-4319-8559-cc874301ffc6)

There are several methods you can use for parameter efficient fine-tuning, each with trade-offs on parameter efficiency, memory efficiency, training speed, model quality, and inference costs.

Let's take a look at the three main classes of PEFT methods. 
- **Selective methods** are those that fine-tune only a subset of the original LLM parameters. There are several approaches that you can take to identify which parameters you want to update. You have the option to train only certain components of the model or specific layers, or even individual parameter types. Researchers have found that the performance of these methods is mixed and there are significant trade-offs between parameter efficiency and compute efficiency.
- **Reparameterization methods** also work with the original LLM parameters, but reduce the number of parameters to train by creating new low rank transformations of the original network weights. A commonly used technique of this type is **LoRA**.
- **Additive methods** carry out fine-tuning by keeping all of the original LLM weights frozen and introducing new trainable components. Here there are two main approaches.
  - **Adapter methods** add new trainable layers to the architecture of the model, typically inside the encoder or decoder components after the attention or feed-forward layers.
  - **Soft prompt methods**, on the other hand, keep the model architecture fixed and frozen, and focus on manipulating the input to achieve better performance. This can be done by adding trainable parameters to the prompt embeddings or keeping the input fixed and retraining the embedding weights. 

## PEFT techniques 1: Low Rank Adoptation of LLM Models (LoRA)
Low-rank Adaptation, or LoRA for short, is a parameter-efficient fine-tuning technique that falls into the **re-parameterization** category. 

Let's take a look at how it works. As a quick reminder, here's the diagram of the transformer architecture that you saw earlier in the course. 
- The input prompt is turned into tokens,
- which are then converted to embedding vectors and
- passed into the encoder and/or decoder parts of the transformer.

In both of these components, there are two kinds of neural networks; self-attention and feedforward networks. The weights of these networks are learned during pre-training. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/8e1d132b-e4f9-4956-8f5d-aab841e01707)

After the embedding vectors are created, they're fed into the self-attention layers where a series of weights are applied to calculate the attention scores. During full fine-tuning, every parameter in these layers is updated.

LoRA is a strategy that reduces the number of parameters to be trained during fine-tuning by freezing all of the original model parameters and then injecting a pair of rank decomposition matrices alongside the original weights. The dimensions of the smaller matrices are set so that their product is a matrix with the same dimensions as the weights they're modifying. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a91d3abb-d6c9-4270-a499-e30501e9782a)

You then keep the original weights of the LLM frozen and train the smaller matrices using the same supervised learning process you saw earlier this week. For inference, the two low-rank matrices are multiplied together to create a matrix with the same dimensions as the frozen weights. You then add this to the original weights and replace them in the model with these updated values. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/8f847f13-23a0-42f8-9135-357465797314)

You now have a LoRA fine-tuned model that can carry out your specific task. Because this model has the same number of parameters as the original, there is little to no impact on inference latency. 

Researchers have found that applying LoRA to just the self-attention layers of the model is often enough to fine-tune for a task and achieve performance gains. 

However, in principle, you can also use LoRA on other components like the feed-forward layers. **But since most of the parameters of LLMs are in the attention layers, you get the biggest savings in trainable parameters by applying LoRA to these weights matrices**. 

Let's look at a practical example using the transformer architecture described in the "Attention is All You Need" paper. 
The paper specifies that the transformer weights have dimensions of 512 by 64. This means that each weights matrix has 32,768 trainable parameters. If you use LoRA as a fine-tuning method with the rank equal to eight, you will instead train two small rank decomposition matrices whose small dimension is eight. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/73f07c3b-a1c5-4c14-99fd-008f8c49c8a7)

This means that Matrix A will have dimensions of 8 by 64, resulting in 512 total parameters. Matrix B will have dimensions of 512 by 8, or 4,096 trainable parameters. By updating the weights of these new low-rank matrices instead of the original weights, you'll be training 4,608 parameters instead of 32,768 and 86% reduction. 

Because LoRA allows you to significantly reduce the number of trainable parameters, you can often perform this method of parameter efficient fine tuning with a single GPU and avoid the need for a distributed cluster of GPUs. Since the rank-decomposition matrices are small, you can fine-tune a different set for each task and then switch them out at inference time by updating the weights. 

Suppose you train a pair of LoRA matrices for a specific task; let's call it Task A. To carry out inference on this task, you would multiply these matrices together and then add the resulting matrix to the original frozen weights. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/6c53482f-d358-4df0-906c-907285ea30fe)

You then take this new summed weights matrix and replace the original weights where they appear in your model. You can then use this model to carry out inference on Task A. If instead, you want to carry out a different task, say Task B, you simply take the LoRA matrices you trained for this task, calculate their product, and then add this matrix to the original weights and update the model again.

The memory required to store these LoRA matrices is very small. So in principle, you can use LoRA to train for many tasks. Switch out the weights when you need to use them, and avoid having to store multiple full-size versions of the LLM. 

How good are these models? Let's use the ROUGE metric you learned about earlier this week to compare the performance of a LoRA fine-tune model to both an original base model and a full fine-tuned version. Let's focus on fine-tuning the FLAN-T5 for dialogue summarization, which you explored earlier in the week.

Just to remind you, the **FLAN-T5-base model has had an initial set of full fine-tuning carried out using a large instruction data set**. First, let's set a baseline score for the FLAN-T5 base model and the summarization data set we discussed earlier. 

Here are the ROUGE scores for the base model where a higher number indicates better performance. You should focus on the ROUGE 1 score for this discussion, but you could use any of these scores for comparison. As you can see, the scores are fairly low. Next, look at the scores for a model that has had additional full fine-tuning on dialogue summarization. 

Remember, although FLAN-T5 is a capable model, it can still benefit from additional fine-tuning on specific tasks. With full fine-tuning, you update every way in the model during supervised learning. You can see that this results in a much higher ROUGE 1 score increasing over the base FLAN-T5 model by 0.19. The additional round of fine-tuning has greatly improved the performance of the model on the summarization task. 

Now let's take a look at the scores for the LoRA fine-tune model. You can see that this process also resulted in a big boost in performance. The ROUGE 1 score has increased from the baseline by 0.17. This is a little lower than full fine-tuning, but not much. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/72f0a5d0-33dc-40fb-87bb-71b1855adb7c)

However, using LoRA for fine-tuning trained a much smaller number of parameters than full fine-tuning using significantly less compute, so this small trade-off in performance may well be worth it. 

You might be wondering how to choose the rank of the LoRA matrices. This is a good question and still an active area of research.
In principle, the smaller the rank, the smaller the number of trainable parameters, and the bigger the savings on compute. However, there are some issues related to model performance to consider. In the paper that first proposed LoRA, researchers at Microsoft explored how different choices of rank impacted the model performance on language generation tasks. 

You can see the summary of the results in the table here. The table shows the rank of the LoRA matrices in the first column, the final loss value of the model, and the scores for different metrics, including BLEU and ROUGE.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/f08a48e9-8ff2-4d15-98e8-a88fec0922ee)

The bold values indicate the best scores that were achieved for each metric. The authors found a plateau in the loss value for ranks greater than 16. 

In other words, **using larger LoRA matrices didn't improve performance**. The takeaway here is that ranks in the range of 4-32 can provide you with a good trade-off between reducing trainable parameters and preserving performance. 

Optimizing the choice of rank is an ongoing area of research and best practices may evolve as more practitioners like you make use of LoRA. 
**LoRA is a powerful fine-tuning method that achieves great performance**. The principles behind the method are useful not just for training LLMs, but for models in other domains. 

# PEFT Techniques 2: Soft prompts
With LoRA, the goal was to find an efficient way to update the weights of the model without having to train every single parameter again. There are also additive methods within PEFT that aim to improve model performance without changing the weights at all. 

We'll explore a second parameter efficient fine tuning method called **prompt tuning**. Now, prompt tuning sounds a bit like prompt engineering, but they are quite different from each other. With prompt engineering, you work on the language of your prompt to get the completion you want. This could be as simple as trying different words or phrases or more complex, like including examples for one or Few-shot Inference. The goal is to help the model understand the nature of the task you're asking it to carry out and to generate a better completion.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/6b80f1b7-f9df-4c8b-9301-50ae29e00e03)

However, there are some limitations to prompt engineering, as it can require a lot of manual effort to write and try different prompts. You're also limited by the length of the context window, and at the end of the day, you may still not achieve the performance you need for your task. With prompt tuning, you add additional trainable tokens to your prompt and leave it up to the supervised learning process to determine their optimal values. The set of trainable tokens is called a **soft prompt**, and it gets prepended to embedding vectors that represent your input text. 

The soft prompt vectors have the same length as the embedding vectors of the language tokens. And including somewhere between 20 and 100 virtual tokens can be sufficient for good performance. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/869f8100-1d8d-4fb9-9fc8-b03841240f5c)

The tokens that represent natural language are hard in the sense that they each correspond to a fixed location in the embedding vector space. However, the soft prompts are not fixed discrete words of natural language. Instead, you can think of them as virtual tokens that can take on any value within the continuous multidimensional embedding space. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/4588e49a-4809-4138-b73a-73c655bb6c51)

However, the soft prompts are not fixed discrete words of natural language. Instead, you can think of them as virtual tokens that can take on any value within the continuous multidimensional embedding space. And through supervised learning, the model learns the values for these virtual tokens that maximize performance for a given task. In full fine tuning, the training data set consists of input prompts and output completions or labels.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/929a677b-1fa7-454e-8354-369d15f99c0e)

The weights of the large language model are updated during supervised learning. In contrast with prompt tuning, the weights of the large language model are frozen and the underlying model does not get updated. Instead, the embedding vectors of the soft prompt gets updated over time to optimize the model's completion of the prompt. Prompt tuning is a very parameter efficient strategy because only a few parameters are being trained. In contrast with the millions to billions of parameters in full fine tuning, similar to what you saw with LoRA.

You can train a different set of soft prompts for each task and then easily swap them out at inference time. You can train a set of soft prompts for one task and a different set for another. To use them for inference, you prepend your input prompt with the learned tokens to switch to another task, you simply change the soft prompt. Soft prompts are very small on disk, so this kind of fine tuning is extremely efficient and flexible. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/8d3e2680-8d0c-45f6-8fe4-21697210ff16)

You'll notice the same LLM is used for all tasks, all you have to do is switch out the soft prompts at inference time. 

So how well does prompt tuning perform? In the original paper, Exploring the Method by Brian Lester and collaborators at Google. The authors compared prompt tuning to several other methods for a range of model sizes. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a66aa697-8682-40a8-9035-fc30559a8ff9)

In this figure from the paper, you can see the Model size on the X axis and the SuperGLUE score on the Y axis. This is the evaluation benchmark you learned about earlier this week that grades model performance on a number of different language tasks. The red line shows the scores for models that were created through full fine tuning on a single task. While the orange line shows the score for models created using multitask fine tuning. The green line shows the performance of prompt tuning and finally, the blue line shows scores for prompt engineering only. 

As you can see, prompt tuning doesn't perform as well as full fine tuning for smaller LLMs. However, as the model size increases, so does the performance of prompt tuning. And once models have around 10 billion parameters, prompt tuning can be as effective as full fine tuning and offers a significant boost in performance over prompt engineering alone. 

One potential issue to consider is the interpretability of learned virtual tokens. Remember, because the soft prompt tokens can take any value within the continuous embedding vector space. The trained tokens don't correspond to any known token, word, or phrase in the vocabulary of the LLM. However, an analysis of the nearest neighbor tokens to the soft prompt location shows that they form tight semantic clusters. In other words, the words closest to the soft prompt tokens have similar meanings. The words identified usually have some meaning related to the task, suggesting that the prompts are learning word like representations.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/54e70de8-2790-40bd-9c2a-d29d3331323a)

# PEFT Method Summary
We explored two PEFT methods in this lesson LoRA, which uses **rank decomposition matrices** to update the model parameters in an efficient way. And Prompt Tuning, where **trainable tokens are added to your prompt and the model weights are left untouched**. Both methods enable you to fine tune models with the potential for improved performance on your tasks while using much less compute than full fine tuning methods. 

LoRA is broadly used in practice because of the comparable performance to full fine tuning for many tasks and data sets.

# Lab 2: Fine-tune a generative AI model for dialogue summarization
https://labs.vocareum.com/main/main.php?m=clabide&mode=s&asnid=1843535&stepid=1843536&hideNavBar=1

```sh
aws s3 cp --recursive s3://dlai-generative-ai/labs/w2-170864/ ./
```
