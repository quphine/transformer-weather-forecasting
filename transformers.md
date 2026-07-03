# Transformers

Unlike traditionally designed LSTMs that works sequentially and takes longer to train on datasets, a Transformer is achitectured in a way that it processes all the data at a time rather than a sequential approach. This significantly reduces training time and efficiency.

## Principle & Methodology

We begin by embedding the tokens from the dataset. In this case, (the Jena climate dataset), every single timestamp entry (with corresponding $14$ features) is considered a token of embedding dim $14$. Our training dataset contains $4320$ input timestamps in order with $144$ labeled temperature predictions (in: $720$ hours, out: temp of next $24$ hours). 
### Embedding
- We cannot consider all these $4320$ timestamps together because, that would need a transformer astronomically big ($\approx18.7M$ attention scores per head per layer ), which is not plausible. 
- Hence we downsample these $4320$ samples step by step using two Conv1D layers to a total of $540$, with the embedding dimension being increased to $128$.

Attention cost scales as: $O(n^{2} \cdot d)$  
$n\equiv$ number of tokens  
$d\equiv$ embedding dim

Inititally we had:
$$C \sim 4320^2\cdot14 = 261.27M$$
After downsampling:
$$C \sim 540^2\cdot128 = 37.32M$$
That is nearly an $85.7\%$ reduction. 

### Multihead Attention
The tokens of dim $(540, 128)$ are multiplied with three special weight matrices. In this implementation, we have 4 attention heads in one Multi Head Block. We also do PreLayerNormalisation to ensure stable training
- Inside an attention head:
    - We the tokens $(540, 128)$ are multiplied with $W_K$, $W_Q$,  $W_V$ weight matrices. These are learnable matrices.
    - The matrices have a dim of $(128, 32)$
    - After multiplying with the respective weight matrices, We obtain $K$(Key), $Q$(Query),  $V$(Value).
    - These matrices are treated as a set of 540 vectors corresponding to each token, and with a embedding dim of 32. 
    - Arrange the $Q$ vectors and $K$ vectors along the rows and columns respectively, and calculate dot products to form the attention score matrix ($A$)
    -  Divide the attention scores $\sqrt{32}$ (Embedding dim).This prevents large dot products from saturating the softmax.
    - Apply softmax along the row such that sum of all attention value for a given query vector is $1$
    - Multiply this $(540, 540)$ matrix with $V$ to obtain the context vectors of size $(540, 32)$

This procedure is folowed for all the 4 attention heads and for comptational efficiency, the weight matrices are fused into one huge $(128, 128)$ matrix rather that 4 separate ones. 

We obtain four context vectors from all the attention heads, which are concatenated side by side to yield an enriched context tensor of size $(540, 128)$

### Encoder block
The encoder block consists of the following:
- 4 attention heads
- Residual Connection
- Layernorm + Dropout 
- Feed forward network ($128 \rightarrow 256 \rightarrow GELU  \rightarrow 128$)
- Dropout
- Residual Connection

Note that the dimension of both the input and the output tensor for this block remains the same.

### Prediction head
After three encoder blocks, the tensors reach the final prediction head, in this implementation, it is a simple MLP. Given the size and highly relevant features of the Jena Climate dataset, the model easily overfits. To prevent this, it is important to keep the param count under control

If the prediction head has a very large parameter count, the model has enough capacity to memorize the dataset and the val loss will skyrocket. 

This MLP takes an input tensor of shape (3,128) and produces 144 temperature predictions. Only the last three token representations are selected for prediction, as each of these vectors has access to the global context through the self-attention mechanism, with the final token naturally incorporating information from the entire input sequence.