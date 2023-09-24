# Model optimizations for deployment
Let's talk about the things you'll have to consider to integrate your model into applications. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a64c88d9-636d-4b78-b84c-9562307f2096)

There are a number of important questions to ask at this stage. The first set is related to how your LLM will function in deployment. So how fast do you need your model to generate completions? What compute budget do you have available? And are you willing to trade off model performance for improved inference speed or lower storage? 

The second set of questions is tied to additional resources that your model may need. Do you intend for your model to interact with external data or other applications? And if so, how will you connect to those resources? 

Lastly, there's the question of how your model will be consumed. What will the intended application or API interface that your model will be consumed through look like? Let's start by exploring a few methods that can be used to optimize your model before deploying it for inference. 

While we could dedicate several lessons to this topic, the aim of this section is to offer you an introduction to the most important optimization techniques. 

Large language models present inference challenges in terms of computing and storage requirements, as well as ensuring low latency for consuming applications. These challenges persist whether you're deploying on premises or to the cloud, and become even more of an issue when deploying to edge devices. One of the primary ways to improve application performance is to reduce the size of the LLM. This can allow for quicker loading of the model, which reduces inference latency.

However, the challenge is to reduce the size of the model while still maintaining model performance. Some techniques work better than others for generative models, and there are tradeoffs between accuracy and performance. You'll learn about three techniques in this section. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/1c157e38-d8e8-4883-a835-dfca396badf8)

Distillation uses a larger model, the teacher model, to train a smaller model, the student model. You then use the smaller model for inference to lower your storage and compute budget. 

Similar to quantization aware training, post training quantization transforms a model's weights to a lower precision representation, such as a 16- bit floating point or eight bit integer. As you learned in week one of the course, this reduces the memory footprint of your model. 

The third technique, Model Pruning, removes redundant model parameters that contribute little to the model's performance. Let's talk through each of these options in more detail.

## Model Distillation
Model Distillation is a technique that focuses on having a **larger teacher model train a smaller student model**. The student model learns to statistically mimic the behavior of the teacher model, either just in the final prediction layer or in the model's hidden layers as well. You'll focus on the first option here. 

You start with your fine tune LLM as your teacher model and create a smaller LLM for your student model. You freeze the teacher model's weights and use it to generate completions for your training data. At the same time, you generate completions for the training data using your student model. The knowledge distillation between teacher and student model is achieved by minimizing a loss function called the **distillation loss**. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/3d1100c0-2be3-458a-b93a-ce7a43ed69f7)

To calculate this loss, distillation uses the probability distribution over tokens that is produced by the **teacher model's softmax layer**. Now, the teacher model is already fine tuned on the training data. So the probability distribution likely closely matches the ground truth data and won't have much variation in tokens. That's why Distillation applies a little trick adding a temperature parameter to the softmax function. 

As we learned in lesson one, a **higher temperature increases the creativity of the language the model generates**. With a temperature parameter greater than one, the probability distribution becomes broader and less strongly peaked. This softer distribution provides you with a set of tokens that are similar to the ground truth tokens. In the context of Distillation, the teacher model's output is often referred to as soft labels and the student model's predictions as soft predictions. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/ee51b185-b41d-4d0e-8253-d6716356776d)

In parallel, you train the student model to generate the correct predictions based on your ground truth training data. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/12094253-2ec1-499b-ae14-156874787bff)

Here, you don't vary the temperature setting and instead use the standard softmax function. Distillation refers to the student model outputs as the hard predictions and hard labels. 

The loss between these two is the student loss. The combined distillation and student losses are used to update the weights of the student model via back propagation. The key benefit of distillation methods is that the smaller student model can be used for inference in deployment instead of the teacher model. 

In practice, distillation is not as effective for generative decoder models. It's **typically more effective for encoder only models**, such as **Burt** that have a lot of representation redundancy. 

**Note that with Distillation, you're training a second, smaller model to use during inference**. You aren't reducing the model size of the initial LLM in any way. 

Let's have a look at the next model optimization technique that actually reduces the size of your LLM. 

## Post-Training Quantization
You were introduced to the second method, quantization, back in week one in the context of training. Specifically Quantization Aware Training, or QAT for short.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/5a6ed97e-8f87-4f4a-b4ef-fb8f0a10d9ba)

However, after a model is trained, you can perform post training quantization, or PTQ for short to optimize it for deployment. PTQ transforms a model's weights to a lower precision representation, such as 16-bit floating point or 8-bit integer. To reduce the model size and memory footprint, as well as the compute resources needed for model serving, quantization can be applied to just the model weights or to both weights and activation layers. 

In general, quantization approaches that include the activations can have a higher impact on model performance. Quantization also requires an extra calibration step to statistically capture the dynamic range of the original parameter values. As with other methods, there are tradeoffs because sometimes quantization results in a small percentage reduction in model evaluation metrics. However, that reduction can often be worth the cost savings and performance gains. 

## Pruning
The last model optimization technique is pruning. At a high level, the goal is to reduce model size for inference by eliminating weights that are not contributing much to overall model performance. These are the weights with values very close to or equal to zero. Note that some pruning methods require full retraining of the model, while others fall into the category of parameter efficient fine tuning, such as LoRA. There are also methods that focus on post-training Pruning. 

In theory, this reduces the size of the model and improves performance. In practice, however, there may not be much impact on the size and performance if only a small percentage of the model weights are close to zero. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/0811055a-0844-4e50-9ee6-cca68b367016)

Quantization, Distillation and Pruning all aim to reduce model size to improve model performance during inference without impacting accuracy. Optimizing your model for deployment will help ensure that your application functions well and provides your users with the best possible experience sense.

# Generative AI Project Lifecycle Cheat Sheet
To help you plan out different stages of the generative AI project life cycle, this cheat sheet provide some indication of the time and effort required for each phase 
of work. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/341de39c-dd32-4b72-846d-b39fc318ef66)

As you saw earlier, pre-training a large language model can be a huge effort. This stage is the most complex you'll face because of the model architecture decisions, the large amount of training data required, and the expertise needed. Remember though, that in general, you will start your development work with an existing foundation model. You'll probably be able to skip this stage. 

If you're working with a foundation model, you'll likely start to assess the model's performance through prompt engineering, which requires less technical expertise, and no additional training of the model. If your model isn't performing as you need, you'll next think about prompt tuning and fine tuning. Depending on your use case, performance goals, and compute budget, the methods you'll try could range from full fine-tuning to parameter efficient fine tuning techniques like laura or prompt tuning. Some level of technical expertise is required for this work. But since fine-tuning can be very successful with a relatively small training dataset, this phase could potentially be completed in a single day. 

Aligning your model using reinforcement learning from human feedback can be done quickly, once you have your train reward model. You'll likely see if you can use an existing reward model for this work, as you saw in this week's lab. However, if you have to train a reward model from scratch, it could take a long time because of the effort involved to gather human feedback. 

Finally, optimization techniques you learned about in the last video, typically fall in the middle in terms of complexity and effort, but can proceed quite quickly assuming the changes to the model don't impact performance too much. After working through all of these steps, you have hopefully trained in tuned a gray LLM that is working well for your specific use case, and is optimized for deployment. 

# Using the LLM in applications
Although all the training, tuning and aligning techniques you've explored can help you build a great model for your application. There are some broader challenges with large language models that can't be solved by training alone. Let's take a look at a few examples. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/f3398b51-21d2-4724-8c92-c3b585adf549)

One issue is that the internal knowledge held by a model cuts off at the moment of pretraining. For example, if you ask a model that was trained in early 2022 who the British Prime Minister is, it will probably tell you Boris Johnson. This knowledge is out of date. The model does not know that Johnson left office in late 2022 because that event happened after its training. 

Models can also struggle with complex math. If you prompt a model to behave like a calculator, it may get the answer wrong, depending on the difficulty of the problem. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a88c442c-c23d-40d9-8f7a-32f154352fb8)

Here, you ask the model to carry out a division problem. The model returns a number close to the correct answer, but it's incorrect. Note the LLMs do not carry out mathematical operations. They are still just trying to predict the next best token based on their training, and as a result, can easily get the answer wrong. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/50ec0d12-1630-4e31-aca0-ec2b7484b1d5)

Lastly, one of the best known problems of LLMs is their tendency to generate text even when they don't know the answer to a problem. This is often called hallucination, and here you can see the model clearly making up a description of a nonexistent plant, the Martian Dunetree. Although there is still no definitive evidence of life on Mars, the model will happily tell you otherwise. 

In this section, we'll learn about some techniques that we can use to help our LLM overcome these issues by connecting to external data sources and applications. We'll have a bit more work to do to be able to connect our LLM to these external components and fully integrate everything for deployment within our application. 

Our application must manage the passing of user input to the large language model and the return of completions. This is often done through some type of orchestration library. This layer can enable some powerful technologies that augment and enhance the performance of the LLM at runtime. By providing access to external data sources or connecting to existing APIs of other applications. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/d35c8a7a-391c-40dd-b7da-5e369c24e733)

One implementation example is Langchain, which you'll learn more about later in this lesson. Let's start by considering how to connect LLMs to external data sources.

## Retrieval Augmented Generation (RAG)
Retrieval Augmented Generation, or RAG for short, is a framework for building LLM powered systems that make use of external data sources. And applications to overcome some of the limitations of these models. RAG is a great way to overcome the knowledge cutoff issue and help the model update its understanding of the world. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/1eee31d0-826e-42b6-805f-7cf1f9da7f9d)

While you could retrain the model on new data, this would quickly become very expensive. And require repeated retraining to regularly update the model with new knowledge. A more flexible and less expensive way to overcome knowledge cutoffs is to **give your model access to additional external data at inference time**. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/ec8ee63a-a3fe-4867-afcf-1b46f3391e2a)

RAG is useful in any case where you want the language model to have access to data that it may not have seen. This could be new information documents not included in the original training data, or proprietary knowledge stored in your organization's private databases. Providing your model with external information, can improve both the relevance and accuracy of its completions. 

Let's take a closer look at how this works. Retrieval augmented generation isn't a specific set of technologies, but rather a framework for providing LLMs access to data they did not see during training. A number of different implementations exist, and the one you choose will depend on the details of your task and the format of the data you have to work with. 

Here you'll walk through the implementation discussed in one of the earliest papers on RAG by researchers at Facebook, originally published in 2020. At the heart of this implementation is a model component called the Retriever, which consists of a query encoder and an external data source.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/5a9fa525-f937-424b-8b5b-f0b7ecea869d)

The encoder takes the user's input prompt and encodes it into a form that can be used to query the data source. In the Facebook paper, the external data is a vector store, which we'll discuss in more detail shortly. But it could instead be a SQL database, CSV files, or other data storage format. These two components are trained together to find documents within the external data that are most relevant to the input query. The Retriever returns the best single or group of documents from the data source and combines the new information with the original user query. The new expanded prompt is then passed to the language model, which generates a completion that makes use of the data.

Let's take a look at a more specific example. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/f76d6531-b5c9-42be-ad52-63a4ccd78b66)

Imagine you are a lawyer using a large language model to help you in the discovery phase of a case. A RAG architecture can help you ask questions of a corpus of documents, for example, previous court filings. Here you ask the model about the plaintiff named in a specific case number.

The prompt is passed to the query encoder, which encodes the data in the same format as the external documents. And then searches for a relevant entry in the corpus of documents. Having found a piece of text that contains the requested information, the Retriever then combines the new text with the original prompt. 

The expanded prompt that now contains information about the specific case of interest is then passed to the LLM. The model uses the information in the context of the prompt to generate a completion that contains the correct answer. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/993c76ee-d393-4c98-9d7d-e47071467b86)

The use case you have seen here is quite simple and only returns a single piece of information that could be found by other means. But imagine the power of RAG to be able to generate summaries of filings or identify specific people, places and organizations within the full corpus of the legal documents. Allowing the model to access information contained in this external data set greatly increases its utility for this specific use case.

In addition to overcoming knowledge cutoffs, RAG also helps you avoid the problem of the model hallucinating when it doesn't know the answer. 

### RAG Integrates with multiple datasources
RAG architectures can be used to integrate multiple types of external information sources. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/0119ef8c-8ae9-43f7-b2b3-8698eba3f6e9)

You can augment large language models with access to local documents, including private wikis and expert systems. Rag can also enable access to the Internet to extract information posted on web pages, for example, Wikipedia. 
By encoding the user input prompt as a SQL query, RAG can also interact with databases. 
Another important data storage strategy is a Vector Store, which contains vector representations of text. This is a particularly useful data format for language models, since internally they work with vector representations of language to generate text. Vector stores enable a fast and efficient kind of relevant search based on similarity. 

## Data Preparation For Vector Store For RAG
Note that implementing RAG is a little more complicated than simply adding text into the large language model. There are a couple of key considerations to be aware of, starting with:
- The size of the context window. Most text sources are too long to fit into the limited context window of the model, which is still at most just a few thousand tokens. Instead, the external data sources are chopped up into many chunks, each of which will fit in the context window. Packages like Langchain can handle this work for you.
  
![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/eb8e8d99-503f-4ed0-b1aa-aeacd40b3d7c)
  
- Second, the data must be available in a format that allows for easy retrieval of the most relevant text. Recall that large language models don't work directly with text, but instead create vector representations of each token in an embedding space.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/c39ee9cc-82bc-4927-9a8d-8639444995db)

These embedding vectors allow the LLM to identify semantically related words through measures such as cosine similarity, which you learned about earlier.
Rag methods take the small chunks of external data and process them through the large language model, to create embedding vectors for each. These new representations of the data can be stored in structures called vector stores, which allow for fast searching of datasets and efficient identification of semantically related text.

## Vector Database Search
Vector databases are a particular implementation of a vector store where each vector is also identified by a key. This can allow, for instance, the text generated by RAG to also include a citation for the document from which it was received. So you've seen how access to external data sources can help a model overcome limits to its internal knowledge. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/c0f43116-04d1-4754-87fa-d2312735b37c)

By providing up to date relevant information and avoiding hallucinations, you can greatly improve the experience of using your application for your users. 

# Interacting with external applications
Now let's take a look at how they can interact with external applications. To motivate the types of problems and use cases that require this kind of augmentation of the LLM, you'll revisit the customer service bot example you saw earlier in the course. 

During this walkthrough of one customer's interaction with ShopBot, you'll take a look at the integrations that you'd need to allow the app to process a return requests from end to end. In this conversation, the customer has expressed that they want to return some Jeans that they purchased. ShopBot responds by asking for the order number, which the customer then provides. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/f07504a5-d4bf-440a-a074-c9d4bcb88489)

ShopBot then looks up the order number in the transaction database. One way it could do this is by using a RAG implementation of the kind you saw earlier. 

In this case here, you would likely be retrieving data through a SQL query to a back-end order database rather than retrieving data from a corpus of documents. Once ShopBot has retrieved the customers order, the next step is to confirm the items that will be returned. The bot ask the customer if they'd like to return anything other than the Jeans.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/22a7dd40-5e27-4b50-8447-4a8106f89f0b)

After the user states their answer, the bot initiates a request to the company's shipping partner for a return label. The body uses the shippers Python API to request the label ShopBot is going to email the shipping label to the customer. It also asks them to confirm their email address. The customer responds with their email address and the bot includes this information in the API call to the shipper. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/59830d4e-8d67-4b1b-8b23-8342c68b6c92)

Once the API request is completed, the Bartlett's the customer know that the label has been sent by email, and the conversation comes to an end. 

This short example illustrates just one possible set of interactions that you might need an LLM to be capable of to power and application. In general, connecting LLMs to external applications allows the model to interact with the broader world, **extending their utility beyond language tasks**. 

As the shop bot example showed, LLMs can be used to trigger actions when given the ability to interact with APIs. LLMs can also connect to other programming resources. For example, a Python interpreter that can enable models to incorporate accurate calculations into their outputs. 

It's important to note that prompts and completions are at the very heart of these workflows. The actions that the app will take in response to user requests will be determined by the LLM, which serves as the application's reasoning engine. In order to trigger actions, the completions generated by the LLM must contain certain important information. 
- First, the model needs to be able to generate a set of instructions so that the application knows what actions to take. These instructions need to be understandable and correspond to allowed actions.
  - In the ShopBot example for instance, the important steps were;
    - checking the order ID,
    - requesting a shipping label,
    - verifying the user email, 
    - And emailing the user the label.
- Second, the completion needs to be formatted in a way that the broader application can understand. This could be as simple as a specific sentence structure or as complex as writing a script in Python or generating a SQL command. For example, here is a SQL query that would determine whether an order is present in the database of all orders.

  ![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/bb40ce36-8073-4359-8c68-6ca21aebe42d)

- Lastly, the model may need to collect information that allows it to validate an action. For example, in the ShopBot conversation, the application needed to verify the email address the customer used to make the original order. Any information that is required for validation needs to be obtained from the user and contained in the completion so it can be passed through to the application.

Structuring the prompts in the correct way is important for all of these tasks and can make a huge difference in the quality of a plan generated or the adherence to a desired output format specification.

# Helping LLMs reason and plan with chain-of-thought
As you saw, it is important that LLMs can reason through the steps that an application must take, to satisfy a user request. Unfortunately, complex reasoning can be challenging for LLMs, especially for problems that involve multiple steps or mathematics. These problems exist even in large models that show good performance at many other tasks. 

Here's one example where an LLM has difficulty completing the task. You're asking the model to solve a simple multi-step math problem, to determine how many apples a cafeteria has after using some to make lunch, and then buying some more. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/897318e5-3f1c-4152-9a3a-498c1c84ea7b)

Your prompt includes a similar example problem, complete with the solution, to help the model understand the task through one-shot inference. If you like, you can pause the video here for a moment and solve the problem yourself. After processing the prompt, the model generates the completion shown here, stating that the answer is 27. This answer is incorrect, as you found out if you solve the problem. The cafeteria actually only has nine apples remaining. **Researchers have been exploring ways to improve the performance of large language models on reasoning tasks**, like the one you just saw.

One strategy that has demonstrated some success is prompting the model to think more like a human, by breaking the problem down into steps. What do I mean by thinking more like a human? Well, here is the one-shot example problem from the prompt on the previous slide. 

The task here is to calculate how many tennis balls Roger has after buying some new ones. One way that a human might tackle this problem is as follows. 
- Begin by determining the number of tennis balls Roger has at the start.
- Then note that Roger buys two cans of tennis balls. Each can contains three balls, so he has a total of six new tennis balls.
- Next, add these 6 new balls to the original 5, for a total of 11 balls.
- Then finish by stating the answer.

These intermediate calculations form the reasoning steps that a human might take, and the full sequence of steps illustrates the chain of thought that went into solving the problem. Asking the model to mimic this behavior is known as chain of thought prompting. It works by including a series of intermediate reasoning steps into any examples that you use for one or few-shot inference. 

By structuring the examples in this way, you're essentially teaching the model how to reason through the task to reach a solution. Here's the same apples problem you saw a couple of slides ago, now reworked as a chain of thought prompt. The story of Roger buying the tennis balls is still used as the one-shot example. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/7f3d01c4-04bd-43bf-a91f-7a3fbb849e16)

But this time you include intermediate reasoning steps in the solution text. These steps are basically equivalent to the ones a human might take, that you saw just a few minutes ago. You then send this chain of thought prompt to the large language model, which generates a completion. 

Notice that the model has now produced a more robust and transparent response that explains its reasoning steps, following a similar structure as the one-shot example. The model now correctly determines that nine apples are left. Thinking through the problem has helped the model come to the correct answer. One thing to note is that while the input prompt is shown here in a condensed format to save space, the entire prompt is actually included in the output. 

You can use chain of thought prompting to help LLMs improve their reasoning of other types of problems too, in addition to arithmetic. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/27cd782b-3174-4fe0-8086-39f32ca9a3a8)

Here's an example of a simple physics problem, where the model is being asked to determine if a gold ring would sink to the bottom of a swimming pool. The chain of thought prompt included as the one-shot example here, shows the model how to work through this problem, by reasoning that a pair would flow because it's less dense than water. When you pass this prompt to the LLM, it generates a similarly structured completion. The model correctly identifies the density of gold, which it learned from its training data, and then reasons that the ring would sink because gold is much more dense than water.

Chain of thought prompting is a powerful technique that improves the ability of your model to reason through problems. While this can greatly improve the performance of your model, the limited math skills of LLMs can still cause problems if your task requires accurate calculations, like totaling sales on an e-commerce site, calculating tax, or applying a discount.

# Program-aided language models (PAL)
the ability of LLMs to carry out arithmetic and other mathematical operations is limited. While you can try using chain of thought prompting to overcome this, it will only get you so far. Even if the model correctly reasons through a problem, it may still get the individual math operations wrong, especially with larger numbers or complex operations. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/2ad46e50-4eac-4baf-9bf4-156f0a18f5c3)

Here's the example you saw earlier where the LLM tries to act like a calculator but gets the answer wrong. 

Remember, the model isn't actually doing any real math here. **It is simply trying to predict the most probable tokens that complete the prompt**. The model getting math wrong can have many negative consequences depending on your use case, like charging customers the wrong total or getting the measurements for a recipe incorrect. 

You can overcome this limitation by **allowing your model to interact with external applications that are good at math, like a Python interpreter**. One interesting framework for augmenting LLMs in this way is called **program-aided language models**, or PAL for short. 

This work first presented by Luyu Gao and collaborators at Carnegie Mellon University in 2022, pairs an LLM with an external code interpreter to carry out calculations. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/56e01862-a6db-470b-91ca-c3b92213889a)

The method makes use of chain of thought prompting to generate executable Python scripts. The scripts that the model generates are passed to an interpreter to execute. The image on the right here is taken from the paper and show some example prompts and completions. 

The strategy behind PAL is to have the LLM generate completions where reasoning steps are accompanied by computer code. This code is then passed to an interpreter to carry out the calculations necessary to solve the problem. You specify the output format for the model by including examples for one or few shot inference in the prompt. 

Let's take a closer look at how these example prompts are structured. You'll continue to work with the story of Roger buying tennis balls as the one-shot example. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/ec03fbe3-17c6-48dd-8f68-c7df678f3c0d)

The setup here should now look familiar. This is a chain of thought example. You can see the reasoning steps written out in words on the lines highlighted in blue. What differs from the prompts you saw before is the inclusion of lines of Python code shown in pink. These lines translate any reasoning steps that involve calculations into code. Variables are declared based on the text in each reasoning step. Their values are assigned either directly, as in the first line of code here, or as calculations using numbers present in the reasoning text as you see in the second Python line. The model can also work with variables it creates in other steps, as you see in the third line. 

Note that the text of each reasoning step begins with a pound sign, so that the line can be skipped as a comment by the Python interpreter. The prompt here ends with the new problem to be solved. In this case, the objective is to determine how many loaves of bread a bakery has left after a day of sales and after some loaves are returned from a grocery store partner. 

On the right, you can see the completion generated by the LLM. Again, the chain of thought reasoning steps are shown in blue and the Python code is shown in pink. As you can see, the model creates a number of variables to track the loaves baked, the loaves sold in each part of the day, and the loaves returned by the grocery store. The answer is then calculated by carrying out arithmetic operations on these variables. The model correctly identifies whether terms should be added or subtracted to reach the correct total. 

Now that you know how to structure examples that will tell the LLM to write Python scripts based on its reasoning steps, let's go over how the PAL framework enables an LLM to interact with an external interpreter. 

To prepare for inference with PAL, you'll format your prompt to contain one or more examples. Each example should contain a question followed by reasoning steps in lines of Python code that solve the problem. Next, you will append the new question that you'd like to answer to the prompt template. Your resulting PAL formatted prompt now contains both the example and the problem to solve. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/09cce2ac-0adf-4a0d-9d26-01c09031651a)

Next, you'll pass this combined prompt to your LLM, which then generates a completion that is in the form of a Python script having learned how to format the output based on the example in the prompt. You can now hand off the script to a Python interpreter, which you'll use to run the code and generate an answer. 

For the bakery example script you saw on the previous slide, the answer is 74. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/b1d7f5aa-1359-4a1d-8182-8f13191cbe8a)

You'll now append the text containing the answer, which you know is accurate because the calculation was carried out in Python to the PAL formatted prompt you started with. By this point you have a prompt that includes the correct answer in context. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/0ae67020-f202-4d80-a25a-331a8307f781)

Now when you pass the updated prompt to the LLM, it generates a completion that contains the correct answer. 

Given the relatively simple math in the bakery bread problem, it's likely that the model may have gotten the answer correct just with chain of thought prompting. But for more complex math, including arithmetic with large numbers, trigonometry or calculus, PAL is a powerful technique that allows you to ensure that any calculations done by your application are accurate and reliable. 

You might be wondering how to automate this process so that you don't have to pass information back and forth between the LLM, and the interpreter by hand. This is where the **orchestrator** that you saw earlier comes in. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/5d1bf2e5-0c34-441f-87e8-8de400686066)

The orchestrator shown here as the yellow box is a technical component that can manage the flow of information and the initiation of calls to external data sources or applications. It can also decide what actions to take based on the information contained in the output of the LLM.

Remember, the LLM is your application's reasoning engine. Ultimately, it creates the plan that the orchestrator will interpret and execute. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/b4f84c96-ca44-405b-b2e8-af19334e80c4)

In PAL there's only one action to be carried out, the execution of Python code. The LLM doesn't really have to decide to run the code, it just has to write the script which the orchestrator then passes to the external interpreter to run. However, most real-world applications are likely to be more complicated than the simple PAL architecture. Your use case may require interactions with several external data sources. As you saw in the shop bot example, you may need to manage multiple decision points, validation actions, and calls to external applications.

# ReAct: Combining reasoning and action
An application making use of PAL can link the LLM to a Python interpreter to run the code and return the answer to the LLM. Most applications will require the LLM to manage more complex workflows, perhaps in including interactions with multiple external data sources and applications. 

We'll explore a framework called **ReAct** that can help LLMs plan out and execute these workflows. **ReAct is a prompting strategy that combines chain of thought reasoning with action planning**. 

The framework was proposed by researchers at Princeton and Google in 2022. The paper develops a series of complex prompting examples based on problems from Hot Pot QA, a multi-step question answering benchmark. That requires reasoning over two or more Wikipedia passages and fever, a benchmark that uses Wikipedia passages to verify facts. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/c6bce2a4-ca7e-4721-b32a-5d2d6ab57d07)

The figure on the right shows some example prompts from the paper, and we'll explore one shortly. 

ReAct uses structured examples to show a large language model how to reason through a problem and decide on actions to take that move it closer to a solution. The example prompts start with a question that will require multiple steps to answer. 

![Uploading image.png…]()

In this example, the goal is to determine which of two magazines was created first. The example then includes a related thought action observation trio of strings. The thought is a reasoning step that demonstrates to the model how to tackle the problem and identify an action to take. 

In the newspaper publishing example, the prompt specifies that the model will search for both magazines and determine which one was published first. 

In order for the model to interact with an external application or data source, it has to identify an action to take from a pre-determined list. In the case of the ReAct framework, the authors created a small Python API to interact with Wikipedia. The three allowed actions are search, which looks for a Wikipedia entry about a particular topic lookup, which searches for a string on a Wikipedia page. And finish, which the model carries out when it decides it has determined the answer. 

As you saw on the previous slide, the thought in the prompt identified two searches to carry out one for each magazine. In this example, the first search will be for Arthur's magazine. The action is formatted using the specific square bracket notation you see here, so that the model will format its completions in the same way. The Python interpreter searches for this code to trigger specific API actions. The last part of the prompt template is the observation, this is where the new information provided by the external search is brought into the context of the prompt. For the model to interpret the prompt then repeats the cycle as many times as is necessary to obtain the final answer. In the second thought, the prompt states the start year of Arthur's magazine and identifies the next step needed to solve the problem. The second action is to search for first for women, and the second observation includes text that states the start date of the publication, in this case 1989.
