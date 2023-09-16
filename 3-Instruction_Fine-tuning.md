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
Multitask fine-tuning is an extension of single task fine-tuning, where the training dataset is comprised of example inputs and outputs for multiple tasks. Here, the dataset contains examples that instruct the model to carry out a variety of tasks, including summarization, review rating, code translation, and entity recognition.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/9dd5f532-554c-46d8-a1f7-4056b7099255)

You train the model on this mixed dataset so that it can improve the performance of the model on all the tasks simultaneously, thus avoiding the issue of catastrophic forgetting. Over many epochs of training, the calculated losses across examples are used to update the weights of the model, resulting in an instruction tuned model that is learned how to be good at many different tasks simultaneously. 

One drawback to multitask fine-tuning is that it requires a lot of data. You may need as many as 50-100,000 examples in your training set. However, it can be really worthwhile and worth the effort to assemble this data. The resulting models are often very capable and suitable for use in situations where good performance at many tasks is desirable.

Let's take a look at one family of models that have been trained using multitask instruction fine-tuning. Instruct model variance differ based on the datasets and tasks used during fine-tuning. 

## Instruction Fine-tuning with FLAN
One example is the FLAN family of models. FLAN, which stands for **fine-tuned language net**, is a specific set of instructions used to fine-tune different models. Because they're FLAN fine-tuning is the last step of the training process the authors of the original paper called it the metaphorical dessert to the main course of pre-training quite a fitting name. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/43e0103e-b12d-4efb-a64b-5daa303e42e5)

FLAN-T5, the FLAN instruct version of the T5 foundation model while FLAN-PALM is the flattening struct version of the palm foundation model. You get the idea, FLAN-T5 is a great general purpose instruct model. In total, **it's been fine tuned on 473 datasets across 146 task categories**. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/125181b0-3d1d-478c-9980-ade16dc3401b)

Those datasets are chosen from other models and papers as shown here. Don't worry about reading all the details right now. If you're interested, you can access the original paper through a reading exercise after the video and take a closer look.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/2092e5b9-d19c-48b2-b6eb-57fd73501fe6)

One example of a prompt dataset used for summarization tasks in FLAN-T5 is SAMSum. It's part of the muffin collection of tasks and datasets and is used to train language models to summarize dialogue. 

### SAMSum: A Dialog Dataset
SAMSum is a dataset with 16,000 messenger like conversations with summaries. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/42c0516e-a35d-4416-803b-0f1ad1064a1c)

Three examples are shown here with the dialogue on the left and the summaries on the right. The dialogues and summaries were crafted by linguists for the express purpose of generating a high-quality training dataset for language models. The linguists were asked to create conversations similar to those that they would write on a daily basis, reflecting their proportion of topics of their real life messenger conversations. 

Although language experts then created short summaries of those conversations that included important pieces of information and names of the people in the dialogue. 

Here is a prompt template designed to work with this SAMSum dialogue summary dataset. The template is actually comprised of several different instructions that all basically ask the model to do this same thing. Summarize a dialogue. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/53cf71e9-4bcd-4230-9307-bb4d96dfb7d5)

For example, briefly summarize that dialogue. What is a summary of this dialogue? What was going on in that conversation?

Including different ways of saying the same instruction helps the model generalize and perform better. Just like the prompt templates you saw earlier. You see that in each case, the dialogue from the SAMSum dataset is inserted into the template wherever the dialogue field appears. The summary is used as the label. After applying this template to each row in the SAMSum dataset, you can use it to fine tune a dialogue summarization task. 

While FLAN-T5 is a great general use model that shows good capability in many tasks. You may still find that it has room for improvement on tasks for your specific use case. For example, imagine you're a data scientist building an app to support your customer service team, process requests received through a chat bot, like the one shown here. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/2c49b08c-c712-42f0-ba5b-027f9e92d3f1)

Your customer service team needs a summary of every dialogue to identify the key actions that the customer is requesting and to determine what actions should be taken in response. 

**The SAMSum dataset gives FLAN-T5 some abilities to summarize conversations**. However, the examples in the dataset are mostly conversations between friends about day-to-day activities and don't overlap much with the language structure observed in customer service chats. You can perform additional fine-tuning of the FLAN-T5 model using a dialogue dataset that is much closer to the conversations that happened with your bot.

This is the exact scenario that we'll explore in the lab. You'll make use of an additional domain specific summarization dataset called dialogsum to improve FLAN-T5's is ability to summarize support chat conversations. This dataset consists of over 13,000 support chat dialogues and summaries. The dialogue some dataset is not part of the FLAN-T5 training data, so the model has not seen these conversations before. Let's take a look at example from dialogsum and discuss how a further round of fine-tuning can improve the model. This is a support chat that is typical of the examples in the dialogsum dataset.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/22faf6f2-baa3-47e5-a9ae-1aefa8c2bc7c)

The conversation is between a customer and a staff member at a hotel check-in desk. The chat has had a template applied so that the instruction to summarize the conversation is included at the start of the text. 

Now, let's take a look at how FLAN-T5 responds to this prompt before doing any additional fine-tuning, note that the prompt is now condensed on the left to give you more room to examine the completion of the model. Here is the model's response to the instruction. You can see that the model does as it's able to identify that the conversation was about a reservation for Tommy. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/792f471a-ecf6-40ea-a3ed-b73bf289331b)

However, it does not do as well as the human-generated baseline summary, which includes important information such as Mike asking for information to facilitate check-in and the models completion has also invented information that was not included in the original conversation. Specifically the name of the hotel and the city it was located in. 

Now let's take a look at how the model does after fine-tuning on the dialogue some dataset, hopefully, you will agree that this is closer to the human-produced summary. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/218ea3be-cb22-4af3-91cd-62b1d6ecce08)

There is no fabricated information and the summary includes all of the important details, including the names of both people participating in the conversation. This example, use the public dialogue, some dataset to demonstrate fine-tuning on custom data. 

In practice, you'll get the most out of fine-tuning by using your company's own internal data. For example, the support chat conversations from your customer support application. This will help the model learn the specifics of how your company likes to summarize conversations and what is most useful to your customer service colleagues.  

One thing you need to think about when fine-tuning is how to evaluate the quality of your models completions. Next you'll learn about several metrics and benchmarks that you can use to determine how well your model is performing and how much better you're fine-tuned version is than the original base model.

https://arxiv.org/abs/2210.11416

# Model evaluation
Let's explore several metrics that are used by developers of large language models that we can use to assess the performance of our own models and compare to other models out in the world. In traditional machine learning, you can assess how well a model is doing by looking at its performance on training and validation data sets where the output is already known. 

You're able to calculate simple metrics such as accuracy, which states the **fraction of all predictions that are correct** because the models are deterministic. But with large language models where the output is non-deterministic and language-based evaluation is much more challenging. 

Take, for example, the sentence, Mike really loves drinking tea. This is quite similar to Mike adores sipping tea. But how do you measure the similarity? Let's look at these other two sentences. Mike does not drink coffee, and Mike does drink coffee. There is only one word difference between these two sentences. However, the meaning is completely different. Now, for humans like us with squishy organic brains, we can see the similarities and differences. But when you train a model on millions of sentences, you need an automated, structured way to make measurements.

**ROUGE** and **BLEU**, are two widely used evaluation metrics for different tasks. 
**ROUGE** or recall oriented under study for jesting evaluation is primarily employed to assess the quality of automatically generated summaries by comparing them to human-generated reference summaries. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/ebf04b25-11da-47d3-89c1-eedf967ab48e)

On the other hand, **BLEU**, or bilingual evaluation understudy is an algorithm designed to evaluate the quality of machine-translated text, again, by comparing it to human-generated translations. Now the word BLEU is French for blue. You might hear people calling this blue but here I'm going to stick with the original BLEU. Before we start calculating metrics. 

## LLM Evaluation - Metrics - Terminology
Let's review some terminology. In the anatomy of language, a unigram is equivalent to a single word. A bigram is two words and n-gram is a group of n-words. Pretty straightforward stuff.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a738ab5a-a043-4d90-a61f-3418d3f842c4)

First, let's look at the ROUGE-1 metric. To do so, let's look at a human-generated reference sentence. It is cold outside and a generated output that is very cold outside. You can perform simple metric calculations similar to other machine-learning tasks using recall, precision, and F1. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/185903dd-d028-48e9-8c70-d5c70bd3f1ef)

The recall metric measures the number of words or unigrams that are matched between the reference and the generated output divided by the number of words or unigrams in the reference. In this case, that gets a perfect score of one as all the generated words match words in the reference.

Precision measures the unigram matches divided by the output size. The F1 score is the harmonic mean of both of these values. These are very basic metrics that only focused on individual words, hence the one in the name, and don't consider the ordering of the words. 

It can be deceptive. It's easily possible to generate sentences that score well but would be subjectively poor. Stop for a moment and imagine that the sentence generated by the model was different by just one word. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/eb9a575d-eb54-43c2-bf12-46f3a42ccafe)

Not, so it is not cold outside. The scores would be the same. 

You can get a slightly better score by taking into account bigrams or collections of two words at a time from the reference and generated sentence. By working with pairs of words you're acknowledging in a very simple way, the ordering of the words in the sentence.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/724c71aa-9911-46f9-a501-c8d08bcdcd56)

By using bigrams, you're able to calculate a **ROUGE-2**. Now, you can calculate the recall, precision, and F1 score using bigram matches instead of individual words. You'll notice that the scores are lower than the **ROUGE-1** scores. With longer sentences, they're a greater chance that bigrams don't match, and the scores may be even lower. Rather than continue on with ROUGE numbers growing bigger to n-grams of three or fours, let's take a different approach. 

Instead, you'll **look for the longest common subsequence present** in both the generated output and the reference output. In this case, the longest matching sub-sequences are, **it is** and **cold outside**, each with a length of two. You can now use the LCS value to calculate the recall precision and F1 score, where the numerator in both the recall and precision calculations is the length of the longest common subsequence, in this case, two. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/19a904b2-80a2-48a5-b42a-41a135a85e00)

Collectively, these three quantities are known as the **Rouge-L** score. As with all of the rouge scores, you need to take the values in context. You can only use the scores to compare the capabilities of models if the scores were determined for the same task. 

For example, summarization. Rouge scores for different tasks are not comparable to one another. As you've seen, a particular problem with simple rouge scores is that it's possible for a bad completion to result in a good score. 

Take, for example, this generated output, "cold, cold, cold, cold". As this generated output contains one of the words from the reference sentence, it will score quite highly, even though the same word is repeated multiple times. The Rouge-1 precision score will be perfect. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/37b7d1c4-b43c-48c7-93b1-83707efa4c32)

One way you can counter this issue is by using a **clipping function** to limit the number of unigram matches to the maximum count for that unigram within the reference. In this case, there is one appearance of cold and the reference and so a modified precision with a clip on the unigram matches results in a dramatically reduced score. However, you'll still be challenged if their generated words are all present, but just in a different order. 

For example, with this generated sentence, **outside cold it is**. This sentence was called perfectly even on the modified precision with the clipping function as all of the words and the generated output are present in the reference. Whilst using a different rouge score can help experimenting with a n-gram size that will calculate the most useful score will be dependent on the sentence, the sentence size, and your use case. 

Note that many language model libraries, for example, Hugging Face, which we used earlier, include implementations of rouge score that we can use to easily evaluate the output of our model. 

The other score that can be useful in evaluating the performance of your model is the **BLEU score**, which stands for bilingual evaluation under study. Just to remind you that BLEU score is **useful for evaluating the quality of machine-translated text**. The score itself is calculated using the average precision over multiple n-gram sizes. Just like the Rouge-1 score that we looked at before, but calculated for a range of n-gram sizes and then averaged.

Let's take a closer look at what this measures and how it's calculated. The BLEU score quantifies the quality of a translation by checking how many n-grams in the machine-generated translation match those in the reference translation. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/97b2d40c-5b41-4dad-ac96-92f4f6167e32)

To calculate the score, you average precision across a range of different n-gram sizes. If you were to calculate this by hand, you would carry out multiple calculations and then average all of the results to find the BLEU score. For this example, let's take a look at a longer sentence so that you can get a better sense of the scores value. The reference human-provided sentence is, I am very happy to say that I am drinking a warm cup of tea. Now, as you've seen these individual calculations in depth when you looked at rouge, I will show you the results of BLEU using a standard library. 

Calculating the BLEU score is easy with pre-written libraries from providers like Hugging Face and I've done just that for each of our candidate sentences. The first candidate is, I am very happy that I am drinking a cup of tea. The BLEU score is 0.495. 
As we get closer and closer to the original sentence, we get a score that is closer and closer to one. Both rouge and BLEU are quite simple metrics and are relatively low-cost to calculate. You can use them for simple reference as you iterate over your models, but you shouldn't use them alone to report the final evaluation of a large language model. 

**Use rouge for diagnostic evaluation of summarization tasks and BLEU for translation tasks**. For overall evaluation of your model's performance, however, you will need to look at one of the **evaluation benchmarks** that have been developed by researchers. 

# Benchmarks
LLMs are complex, and simple evaluation metrics like the rouge and blur scores, can only tell you so much about the capabilities of your model. In order to measure and compare LLMs more holistically, you can make use of pre-existing datasets, and associated benchmarks that have been established by LLM researchers specifically for this purpose.

Selecting the right evaluation dataset is vital, so that you can accurately assess an LLM's performance, and understand its true capabilities. You'll find it useful to select datasets that isolate specific model skills, like reasoning or common sense knowledge, and those that focus on potential risks, such as disinformation or copyright infringement. An important issue that you should consider is **whether the model has seen your evaluation data during training**.

You'll get a more accurate and useful sense of the model's capabilities by evaluating its performance on data that it hasn't seen before. Benchmarks, such as GLUE, SuperGLUE, or Helm, cover a wide range of tasks and scenarios. They do this by designing or collecting datasets that test specific aspects of an LLM. 

## General Language Understanding Evaluation (GLUE)
GLUE, or General Language Understanding Evaluation, was introduced in 2018. GLUE is a collection of natural language tasks, such as sentiment analysis and question-answering. GLUE was created to encourage the development of models that can generalize across multiple tasks, and you can use the benchmark to measure and compare the model performance. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/17838497-6867-4f14-b685-e4bfff995499)

## Super GLUE
As a successor to GLUE, SuperGLUE was introduced in 2019, to address limitations in its predecessor. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/566802a4-3519-4bd6-8fd6-9bbfecc4dac8)

It consists of a series of tasks, some of which are not included in GLUE, and some of which are more challenging versions of the same tasks. SuperGLUE includes tasks such as multi-sentence reasoning, and reading comprehension. Both the GLUE and SuperGLUE benchmarks have leaderboards that can be used to compare and contrast evaluated models.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/369b07df-d280-4faa-a2dc-a3487a20ab20)

The results page is another great resource for tracking the progress of LLMs. 

As models get larger, their performance against benchmarks such as SuperGLUE start to match human ability on specific tasks. That's to say that models are able to perform as well as humans on the benchmarks tests, but subjectively we can see that they're not performing at human level at tasks in general. There is essentially an arms race between the emergent properties of LLMs, and the benchmarks that aim to measure them. 

Here are a couple of recent benchmarks that are pushing LLMs further. **Massive Multitask Language Understanding**, or MMLU, is designed specifically for modern LLMs. To perform well models must possess extensive world knowledge and problem-solving ability. Models are tested on elementary mathematics, US history, computer science, law, and more. In other words, tasks that extend way beyond basic language understanding. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/aa038b57-8fbd-464c-b2f4-b07006dd4960)

BIG-bench currently consists of 204 tasks, ranging through linguistics, childhood development, math, common sense reasoning, biology, physics, social bias, software development and more. BIG-bench comes in three different sizes, and part of the reason for this is to keep costs achievable, as running these large benchmarks can incur large inference costs. 

A final benchmark you should know about is the **Holistic Evaluation of Language Models**, or **HELM**.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/e79f0941-f224-4506-9999-ffe3b4c3984d)

The HELM framework aims to improve the transparency of models, and to offer guidance on which models perform well for specific tasks. HELM takes a **multimetric approach**, measuring seven metrics across 16 core scenarios, ensuring that trade-offs between models and metrics are clearly exposed. 

One important feature of HELM is that it assesses on **metrics beyond basic accuracy measures**, like **precision of the F1 score**. The benchmark also includes metrics for **fairness, bias, and toxicity**, which are becoming increasingly important to assess as LLMs become more capable of human-like language generation, and in turn of exhibiting potentially harmful behavior. 

HELM is a living benchmark that aims to continuously evolve with the addition of new scenarios, metrics, and models. You can take a look at the results page to browse the LLMs that have been evaluated, and review scores that are pertinent to your project's needs.
