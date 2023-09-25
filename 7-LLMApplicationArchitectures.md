Let's look at the building blocks for creating LLM powered applications. 

# Building Generative Applications 
You'll require several key components to create end-to-end solutions for your applications, starting with the infrastructure layer. 

This layer provides the compute, storage, and network to serve up your LLMs, as well as to host your application components. You can make use of your on-premises infrastructure for this or have it provided for you via on-demand and pay-as-you-go Cloud services. 

Next, you'll include the large language models you want to use in your application. These could include foundation models, as well as the models you have adapted to your specific task. The models are deployed on the appropriate infrastructure for your inference needs. Taking into account whether you need real-time or near-real-time interaction with the model. 

You may also have the need to retrieve information from external sources, such as those discussed in the retrieval augmented generation section. 

Your application will return the completions from your large language model to the user or consuming application. Depending on your use case, you may need to implement a mechanism to capture and store the outputs. For example, you could build the capacity to store user completions during a session to augment the fixed contexts window size of your LLM. 

You can also gather feedback from users that may be useful for additional fine-tuning, alignment, or evaluation as your application matures. Next, you may need to use additional tools and frameworks for large language models that help you easily implement some of the techniques discussed in this course. 

As an example, you can use LangChain's built-in libraries to implement techniques like PAL, REACT or chain of thought prompting. You may also utilize model hubs which allow you to centrally manage and share models for use in applications. 

In the final layer, you typically have some type of user interface that the application will be consumed through, such as a website or a rest API. This layer is where you'll also include the security components required for interacting with your application. 

At a high level, this architecture stack represents the various components to consider as part of your generative AI applications. 

Your users, whether they are human end-users or other systems that access your application through its APIs, will interact with this entire stack. 

# Summary
As you can see, the model is typically only one part of the story in building end-to-end generative AI applications. In summary:

We saw how to align your models with human preferences, such as helpfulness, harmlessness, and honesty by fine-tuning using a technique called **reinforcement learning with human feedback**, or RLHF for short. Given the popularity of RLHF, there are many existing RL reward models and human alignment datasets available, enabling you to quickly start aligning your models. 

In practice, RLHF is a very effective mechanism that you can use to improve the alignment of your models, reduce the toxicity of their responses, and let you use your models more safely in production. You also saw important techniques to optimize your model for inference by reducing the size of the model through distillation, quantization, or pruning. This minimizes the amount of hardware resources needed to serve your LLMs in production. 

Lastly, we explored ways that we can help our model perform better in deployment through structured prompts and connections to external data sources and applications. LLMs can play an amazing role as the reasoning engine in an application, exploiting their intelligence to power exciting, useful applications. Frameworks like LangChain are making it possible to quickly build, deploy, and test LLM powered applications, and it's a very exciting time for developers. 

# Amazon Sagemaker JumpStart
Amazon Sagemaker JumpStart, can help us get into production quickly and operate at scale.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/8e846c92-d203-4471-95c1-d4626c904e0c)

Here's the application stack that we explored previously. As we saw, building an LLM-powered application requires multiple components. 

Sagemaker JumpStart is a model hub, and it allows you to quickly deploy foundation models that are available within the service, and integrate them into your own applications. The JumpStart service also provides an easy way to fine-tune and deploy models. JumpStart covers many parts of this diagram, including the infrastructure, the LLM itself, the tools and frameworks, and even an API to invoke the model. 

In contrast to the models that we worked with in the labs, JumpStart models require GPUs to fine tune and deploy. And keep in mind, these GPUs are subject to on-demand pricing and you should refer to the Sagemaker pricing page before selecting the compute you want to use. Also, please be sure to delete the Sagemaker model endpoints when not in use and follow cost monitoring best practices to optimize cost. 

Sagemaker JumpStart is accessible from the AWS console, or through Sagemaker studio. For this brief tour, I'll start in studio then choose JumpStart from the main screen. I could optionally choose JumpStart from the left-hand menu, and select models, notebooks, and solutions as well. After I click on "JumpStart", you'll see different categories that include end-to-end solutions across different use cases, as well as a number of foundation models for different modalities that you can easily deploy, as well as fine-tune, where yes is indicated under the fine-tuning option. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/1c806aa0-eaf6-4ef6-bce7-c0ba00bf61c3)

Let's look at an example we're all familiar with after working through the course, which is the Flan-T5 model. We've specifically been using the base variant in the course to minimize the resources needed by the lab environments. 

However, as you can see here, you can also utilize other variants of Flan-T5 through JumpStart depending on your needs. You'll also notice the Hugging Face logo here, which means they're actually coming directly from Hugging Face. And AWS has worked with Hugging Face to the point where you can easily, with just a few clicks, deploy, or fine-tune the model. 

In addition to acting as a model hub that includes foundation models, JumpStart also provides a lot of resources in terms of blogs, videos, and example notebooks as well.

# References
# Reinforcement Learning from Human-Feedback (RLHF)
- [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf) - Paper by OpenAI introducing a human-in-the-loop process to create a model that is better at following instructions (**InstructGPT**).
- [Learning to summarize from human feedback](https://arxiv.org/pdf/2009.01325.pdf) - This paper presents a method for improving language model-generated summaries using a reward-based approach, surpassing human reference summaries.

# Proximal Policy Optimization (PPO)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) - The paper from researchers at OpenAI that first proposed the PPO algorithm. The paper discusses the performance of the algorithm on a number of benchmark tasks including robotic locomotion and game play.
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf) - This paper presents a simpler and effective method for precise control of large-scale unsupervised language models by aligning them with human preferences.

# Scaling human feedback
[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073.pdf) - This paper introduces a method for training a harmless AI assistant without human labels, allowing better control of AI behavior with minimal human input.

# Advanced Prompting Techniques
- [Chain-of-thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf) -  Paper by researchers at Google exploring how chain-of-thought prompting improves the ability of LLMs to perform complex reasoning.
- [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435) - This paper proposes an approach that uses the LLM to read natural language problems and generate programs as the intermediate reasoning steps.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - This paper presents an advanced prompting technique that allows an LLM to make decisions about how to interact with external applications.

# LLM powered application architectures
- [LangChain Library (GitHub)](https://github.com/hwchase17/langchain) - This library is aimed at assisting in the development of those types of applications, such as Question Answering, Chatbots and other Agents. You can read the documentation [here](https://docs.langchain.com/docs/).
- [Who Owns the Generative AI Platform?](https://a16z.com/2023/01/19/who-owns-the-generative-ai-platform/) - The article examines the market dynamics and business models of generative AI.

# Active Areas of Research
With the growth of AI comes the recognition that we must all use it responsibly. What are some of the new risks and challenges of responsible AI, specifically in the context of generative AI with large language models?

That's a great question because there are lots of challenges. Let's focus on three. 
- One is toxicity
- Another are the hallucinations
- And a third is the intellectual property issue.

Toxicity at its core, meaning toxic, implies certain language or content that can be harmful or discriminatory towards certain groups, especially towards marginalized groups or protected groups. And so one thing that we can do is start with the training data. As you know, that is the basis of every AI. So you can start with curating the training data. You can also train guardrail models to detect and filter out any unwanted content in the training data. We also think about how much human annotation is involved when it comes to training data and training annotations. We want to make sure we provide enough guidance to those annotators and also have a very diverse group of annotators that we're educating so that they can understand how to pull out certain data or how to mark certain data.

Hallucinations, we think about things that are simply not true, or maybe something that seems like it could be true, but it isn't. It has no basis to it. Well, this is exactly what it means in this case with generative AI, due to the nature of how we train large language models or just neural networks in general. A lot of times we don't know what the model is actually learning and so sometimes what the model will try to do is fill gaps where it has missing data. And oftentimes this leads to false statements or the hallucinations. 

And so one thing that we can do, we can educate the users that this is the reality of this technology and add any disclaimer so they can know that this is something you should be able to look out for. Also, you can augment large language models with independent and verified sources so you can double check against the data that you're getting back. You also want to make sure that you develop methods for attributing generated output to particular pieces of training data so that we can always trace back to where we got the information. And last but not least, we always want to make sure we define what the intended use case is for versus what the unintended use case is.

Intellectual property basically means where people are using data that is returned back from these models in terms of AI. And it can be plagiarizing someone's previous work, or you can have copyright issues for pieces of work and content that already exists. 

And so this is likely to be addressed over time by a mixture of not just the technologies, but also with policymakers and other legal mechanisms. 
Also, we want to incorporate a system of governance to make sure that every stakeholder is doing what they need to do to prevent this from happening in the near term. There's a new concept of machine unlearning in which protected content or its effects on generative AI outputs are reduced or removed. 

So this is just one approach that is very primitive in research today. We can also do filtering or blocking approaches that compare generated content to protected content and training data and suppress or replace it if it's too similar before presenting it to the user. 

How can I responsibly, build and use generative AI models? 
Defining use cases is very important. The more specific, the more narrow the better. 
One example where we actually use gentive AI to test and evaluate the robustness of a system is when it comes to face ID systems. We actually use journeys of AI to create different versions of a face. For example, if I'm trying to test a system that uses my face to unlock my phone, I want to make sure I test it with different versions of my face, with long hair, with short hair, with glasses on, with makeup on, with no makeup on. And we can use gentle AI to do this at scale. And so this is an example of how we use that to test the robustness. 

Also, we want to make sure we access the risk because each use case has its own set of risks. Some may be better or worse. 

Also, evaluating the performance is truly a function of the data and the system. You may have the same system, but when tested with different types of data, may perform very well or may perform very terribly. 

Also, we want to make sure we iterate over the AI lifecycle. It's never a one and done. Creating AI is a continuous Iterative cycle where we want to implement responsibility at the concept stage as well as the deployment stage and monitoring that feedback over time. 

And last but not least, we want to issue governance policies throughout the lifecycle and accountability measures for every stakeholder involved. 

What are some of the topics that the research community is actively working on right now that you find exciting?
There are lots, which is why, again, this field is unraveling every day. 
- There's water marking and fingerprinting which are ways to include almost like a stamp or a signature in a piece of content or data so that we can always trace back. - I think also creating models that help determine if content was created with generative AI is also a budding field of research.

