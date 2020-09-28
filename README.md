# dendric-relu
- The project is mainly a MNIST-set based training playground on `PyTorch` with various loss options (VAE loss, focal loss, GE2E loss). The model consists of a VAE and a classifier network.
- The project also serves to simulate some academic ideas, such as a guess-net based border training enhancement, or compound ReLUs.
- The project supports data parallel training using `DistributedDataParallel` library.

### Guess-Net based Border Training Enhancement
#### Problem
- Another part of ideas in this project was to test out the question: "How can the network more focus on the harder cases on the border?"
- Conceptually, there already exists 'focal loss' solution for this, but the loss basically uses all the training set in the entire training period with the majority of well-classified samples which become gradually less effective and insignificant for the performance attribution in the training.

#### Proposal
- The main idea is to reserve some specific part of network for the harder problems on the border.
- A guess network is proposed which is trained to tell which samples are about to lead a miss-classification, and then it trains the pre-reserved part of network mostly focused on the miss-classification samples.

#### How it works
- In details, the network resource is separated in half, then one half is used for the first tens of epochs for the entire training set. It trains both the original network and the guess network.
- After that, the miss-classified samples on the border are filtered, and then the other half reserved resource is now used to mostly learn the border difference in details, so that more resource can focus on the harder problems.

#### Simulation & Evaluation
- currently ongoing (TBU)

### Compound ReLUs (deprecated)
- This project initially aimed to carry out research on the impact of compound ReLUs 
  which are partially hyper-activating, hypo-activating or standard. However, positioning non-linear activations in the middle layer of network is less efficiently without further compensating measure.
- Also, if they are chunked by middle layer's non-linear activations, it is mostly the same as ResNet
- Hence, there was no point of continuing this and the original project is canceled.
