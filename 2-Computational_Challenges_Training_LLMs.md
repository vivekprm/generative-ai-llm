One of the most common issues you still counter when you try to train large language models is running out of memory. If you've ever tried training or even just loading your model on Nvidia GPUs, this error message might look familiar. 

CUDA, short for **Compute Unified Device Architecture**, is a collection of libraries and tools developed for Nvidia GPUs. Libraries such as PyTorch and TensorFlow use CUDA to boost performance on metrics multiplication and other operations common to deep learning.

You'll encounter these out-of-memory issues because most LLMs are huge, and require a ton of memory to store and train all of their parameters. Let's do some quick math to develop intuition about the scale of the problem. 

A single parameter is typically represented by a 32-bit float, which is a way computers represent real numbers. You'll see more details about how numbers gets stored in this format shortly. A 32-bit float takes up four bytes of memory. So to store one billion parameters you'll need four bytes times one billion parameters, or four gigabyte of GPU RAM at 32-bit full precision. This is a lot of memory, and note, if only accounted for the memory to store the model weights so far.

If you want to train the model, you'll have to plan for additional components that use GPU memory during training. These include two Adam optimizer states, gradients, activations, and temporary variables needed by your functions. This can easily lead to 20 extra bytes of memory per model parameter. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/53943b3b-1d93-490a-9bb2-2b5916d65325)

In fact, to account for all of these overhead during training, you'll actually require approximately 20 times the amount of GPU RAM that the model weights alone take up. To train a one billion parameter model at 32-bit full precision, you'll need approximately 80 gigabyte of GPU RAM. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/5c00d4c2-84f8-4235-ade2-6c31ff3eab54)

This is definitely too large for consumer hardware, and even challenging for hardware used in data centers, if you want to train with a single processor. Eighty gigabyte is the memory capacity of a single **Nvidia A100 GPU**, a common processor used for machine learning tasks in the Cloud. 

What options do you have to reduce the memory required for training? 
One technique that you can use to reduce the memory is called **quantization**. The main idea here is that you reduce the memory required to store the weights of your model by reducing their precision from 32-bit floating point numbers to 16-bit floating point numbers, or eight-bit integer numbers. The corresponding data types used in deep learning frameworks and libraries are **FP32** for 32-bit full position, **FP16**, or **Bfloat16** for 16-bit half precision, and **int8** eight-bit integers. The range of numbers you can represent with FP32 goes from approximately ```3*10^-38``` to ```3*10^38```. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/9b860f06-ebfe-4a6a-9930-b317fa08d2ef)

By default, model weights, activations, and other model parameters are stored in FP32. Quantization statistically projects the original 32-bit floating point numbers into a lower precision space, using scaling factors calculated based on the range of the original 32-bit floating point numbers.

Let's look at an example. Suppose you want to store a PI to six decimal places in different positions. Floating point numbers are stored as a series of bits zeros and ones. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/750fd574-2460-4f8d-9266-a281d8053a48)

The 32 bits to store numbers in full precision with FP32 consist of one bit for the sign where zero indicates a positive number, and one a negative number. Then eight bits for the exponent of the number, and 23 bits representing the fraction of the number. The fraction is also referred to as the mantissa, or significant. It represents the precision bits off the number. If you convert the 32-bit floating point value back to a decimal value, you notice the slight loss in precision. For reference, here's the real value of Pi to 19 decimal places. 

Now, let's see what happens if you project this FP32 representation of Pi into the FP16, 16-bit lower precision space. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/b09f8430-ac9e-4f1a-8a40-0eac947ec3ea)

The 16 bits consists of one bit for the sign, as you saw for FP32, but now FP16 only assigns five bits to represent the exponent and 10 bits to represent the fraction. Therefore, the range of numbers you can represent with FP16 is vastly smaller from negative 65,504 to positive 65,504. The original FP32 value gets projected to 3.140625 in the 16-bit space. Notice that you lose some precision with this projection. There are only six places after the decimal point now. You'll find that this loss in precision is acceptable in most cases because you're trying to optimize for memory footprint.
Storing a value in FP32 requires four bytes of memory. In contrast, storing a value on FP16 requires only two bytes of memory, so with quantization you have reduced the memory requirement by half. 

The AI research community has explored ways to optimize16-bit quantization. One datatype in particular **BFLOAT16**, has recently become a popular alternative to **FP16**. BFLOAT16, short for **Brain Floating Point Format** developed at Google Brain has become a popular choice in deep learning. Many LLMs, including **FLAN-T5**, have been pre-trained with BFLOAT16. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/0afe063c-089e-4e52-8bd9-069a05558533)

**BFLOAT16 or BF16 is a hybrid between half precision FP16 and full precision FP32**. BF16 significantly helps with training stability and is supported by newer GPU's such as NVIDIA's A100. BFLOAT16 is often described as a truncated 32-bit float, as it captures the full dynamic range of the full 32-bit float, that uses only 16-bits. BFLOAT16 uses the full eight bits to represent the exponent, but truncates the fraction to just seven bits. This not only saves memory, but also increases model performance by speeding up calculations. The downside is that BF16 is not well suited for integer calculations, but these are relatively rare in deep learning.

For completeness let's have a look at what happens if you quantize Pi from the 32-bit into the even lower precision eight bit space. If you use one bit for the sign INT8 values are represented by the remaining seven bits. This gives you a range to represent numbers from negative 128 to positive 127 and unsurprisingly Pi gets projected two or three in the 8-bit lower precision space. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/17096873-cdbd-4004-991b-33e59674f6c8)

This brings new memory requirement down from originally four bytes to just one byte, but obviously results in a pretty dramatic loss of precision. 

# Quantization Summary
Let's summarize what you've learned here and emphasize the key points you should take away from this discussion. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/eaa80dc4-c989-4255-b0cd-d573e6dc585e)

Remember that the goal of quantization is to reduce the memory required to store and train models by reducing the precision off the model weights. Quantization statistically projects the original 32-bit floating point numbers into lower precision spaces using scaling factors calculated based on the range of the original 32-bit floats. **Modern deep learning frameworks and libraries support quantization-aware training, which learns the quantization scaling factors during the training process**.

BFLOAT16 has become a popular choice of precision in deep learning as it maintains the dynamic range of FP32, but reduces the memory footprint by half. Many LLMs, including FLAN-T5, have been pre-trained with BFOLAT16. 

# Impact of Quantization
Now let's return to the challenge of fitting models into GPU memory and take a look at the impact quantization can have. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/22fc6f84-27e4-4d24-a6ba-2ec3bbe7388b)

By applying quantization, you can reduce your memory consumption required to store the model parameters down to only two gigabyte using 16-bit half precision of 50% saving and you could further reduce the memory footprint by another 50% by representing the model parameters as eight bit integers, which requires only one gigabyte of GPU RAM. 

Note that in all these cases you still have a model with one billion parameters. As you can see, the circles representing the models are the same size. Quantization will give you the same degree of savings when it comes to training. As you heard earlier, you'll quickly hit the limit of a single NVIDIA A100 GPU with 80 gigabytes of memory. When you try to train a one billion parameter model at 32-bit full precision, you'll need to consider using either 16-bit or eight bit quantization if you want to train on a single GPU.

And remember, many models now have sizes in excess of 50 billion or even 100 billion parameters. Meaning you'd need up to 500 times more memory capacity to train them, tens of thousands of gigabytes. These enormous models dwarf the one billion parameter model we've been considering, shown here to scale on the left.

**As modal scale beyond a few billion parameters, it becomes impossible to train them on a single GPU**. Instead, **you'll need to turn to distributed computing techniques** while you train your model across multiple GPUs. This could require access to hundreds of GPUs, which is very expensive. **Another reason why you won't pre-train your own model from scratch most of the time**. 

However, an additional training process called **fine-tuning**, which you'll learn about next week. Also require storing all training parameters in memory and it's very likely you'll want to fine tune a model at some point.

# Efficient multi-GPU compute strategies
It's very likely that at some point you will need to scale your model training efforts beyond a single GPU.

Even if your model does fit onto a single GPU, there are benefits to using multiple GPUs to speed up your training. It'll find it useful to know how to distribute compute across GPUs even when you're working with a small model.

Let's discuss how we can carry out this scaling across multiple GPUs in an efficient way. We'll begin by considering the case where our model still fits on a single GPU. The first step in scaling model training is to distribute large data-sets across multiple GPUs and process these batches of data in parallel. A popular implementation of this model replication technique is Pi torches distributed data-parallel, or DDP for short. 

## Distributed Data Parallel (DDP)
DDP copyists your model onto each GPU and sends batches of data to each of the GPUs in parallel. Each data-set is processed in parallel and then a synchronization step combines the results of each GPU, which in turn updates the model on each GPU, which is always identical across chips. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/8166c964-c7b5-424a-955f-71cf1eafafc9)

This implementation allows parallel computations across all GPUs that results in faster training. Note that **DDP requires that your model weights and all of the additional parameters, gradients, and optimizer states that are needed for training, fit onto a single GPU**.

If your model is too big for this, you should look into another technique called modal sharding. A popular implementation of modal sharding is Pi Torch is **fully sharded data parallel, or FSDP for short**.

## Fully Sharded Data Parallel (FSDP)
FSDP is motivated by a paper published by researchers at Microsoft in 2019 that proposed a technique called ZeRO. **ZeRO stands for zero redundancy optimizer** and the goal of ZeRO is to optimize memory by distributing or sharding model states across GPUs with **ZeRO data overlap**. 

This allows you to scale model training across GPUs when your model doesn't fit in the memory of a single chip. Let's take a quick look at how ZeRO works before coming back to FSDP. 

We looked at all of the memory components required for training LLMs, the largest memory requirement was for the optimizer states, which take up twice as much space as the weights, followed by weights themselves and the gradients. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/7b0d6c9b-b92c-474f-87f9-a93f8eaa2672)

Let's represent the parameters as this blue box, the gradients and yellow and the optimizer states in green. One limitation off the model replication strategy that I showed before is that you need to keep a full model copy on each GPU, which leads to redundant memory consumption. You are storing the same numbers on every GPU.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/6db5921b-adc4-4250-a734-f1351408f9bd)

### Zero Redundancy Optimizer (ZeRO)
ZeRO eliminates this redundancy by distributing also referred to as sharding the model parameters, gradients, and optimizer states across GPUs instead of replicating them. At the same time, the communication overhead for a sinking model states stays close to that of the previously discussed DDP. ZeRO offers three optimization stages.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/26d0d53e-f0e8-4071-bbc6-c691691496f8)

- ZeRO Stage 1, shards only optimizer states across GPUs, this can reduce your memory footprint by up to a factor of four. 
- ZeRO Stage 2 also shards the gradients across chips. When applied together with Stage 1, this can reduce your memory footprint by up to eight times.
- Finally, ZeRO Stage 3 shards all components including the model parameters across GPUs. When applied together with Stages 1 and 2, memory reduction is linear with a number of GPUs. For example, sharding across 64 GPUs could reduce your memory by a factor of 64.

Let's apply this concept to the visualization of DDP and replace the LLM by the memory representation of model parameters, gradients, and optimizer states. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/333e1e11-6ea8-4c74-b76a-9e6a375edc1a)

When you use FSDP, you distribute the data across multiple GPUs as you saw happening in DDP. But with FSDP, you also distributed or shard the model parameters, gradients, and optimize the states across the GPU nodes using one of the strategies specified in the ZeRO paper. With this strategy, you can now work with models that are too big to fit on a single chip. 

In contrast to DDP, where each GPU has all of the model states required for processing each batch of data available locally, FSDP requires you to collect this data from all of the GPUs before the forward and backward pass. Each CPU requests data from the other GPUs on-demand to materialize the sharded data into unsharded data for the duration of the operation. After the operation, you release the unsharded non-local data back to the other GPUs as original sharded data. You can also choose to keep it for future operations during backward pass for example. Note, this requires more GPU RAM again, this is a typical performance versus memory trade-off decision. 

In the final step after the backward pass, FSDP synchronizes the gradients across the GPUs in the same way they were for DDP. 

Model sharding as described with FSDP allows you to:
- Reduce your overall GPU memory utilization.
- Optionally, you can specify that FSDP **offloads part of the training computation to CPUs** to further reduce your GPU memory utilization.
- To manage the trade-off between performance and memory utilization, you can configure the level of sharding using FSDP is **sharding factor**.

A sharding factor of one basically removes the sharding and replicates the full model similar to DDP. If you set the sharding factor to the maximum number of available GPUs, you turn on full sharding. This has the most memory savings, but **increases the communication volume between GPUs**. Any sharding factor in-between enables hyper sharding. 

Let's take a look at how FSDP performs in comparison to DDP measured in teraflops per GPU.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/3d6ae041-027f-4c24-84a7-59dbe4508224)

These tests were performed using a maximum of **512 NVIDIA V100 GPUs**, each with 80 gigabytes of memory. 

Note, **one teraflop corresponds to one trillion floating-point operations per second**. 

The first figure shows FSDP performance for different size T5 models. You can see the different performance numbers for FSDP, full sharding in blue, hyper shard in orange and full replication in green. For reference, DDP performance is shown in red. 

For the first 25 models with 611 million parameters and 2.28 billion parameters, the performance of FSDP and DDP is similar. Now, if you choose **a model size beyond 2.28 billion, such as T5 with 11.3 billion parameters, DDP runs into the out-of-memory error**. 

FSDP on the other hand can easily handle models this size and achieve much higher teraflops when lowering the model's precision to 16-bit. 

The second figure shows 7% decrease in per GPU teraflops when increasing the number of GPUs from 8-512 for the 11 billion T5 model, plotted here using a batch size of 16 and orange and a batch size of eight in blue. As the model grows in size and is distributed across more and more GPUs, the increase in communication volume between chips starts to impact the performance, slowing down the computation. 

In summary, this shows that you can use FSDP for both small and large models and seamlessly scale your model training across multiple GPUs.

# Scaling Laws and Compute Optimal Models
Here we'll learn about research that has explored the relationship between model size, training, configuration and performance in an effort to determine just how big models need to be. 

Remember, the goal during pre-training is to maximize the model's performance of its learning objective, which is minimizing the loss when predicting tokens.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/3c074115-36ec-4b44-a8e9-a7383ef16865)

Two options you have to achieve better performance are **increasing the size of the dataset** you train your model on and **increasing the number of parameters** in your model. In theory, you could scale either of both of these quantities to improve performance. 

However, another issue to take into consideration is your compute budget which includes factors like the number of GPUs you have access to and the time you have available for training models. 
