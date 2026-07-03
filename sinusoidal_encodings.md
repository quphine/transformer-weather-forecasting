# Sinusoidal Encodings
We have 864 tokens with an embedding dimension of 128 elements each. Our aim is to inject positional information to these embeddings to make the transformer aware of the true positions.

The standard Sinusoidal Embeddings are represented in a matrix the same size as that of the embeddings ($864\times128$) that we add to the original embedding matrix.

## Methodology
- The embedding matrix is grouped into pairs column wise (The first and the second values or indices 0 and 1 of all tokens are a pair, the indices 2 and 3 are a pair and so on). In our case, we have $\frac{128}{2}=64$ such pairs per token under consideration

- Now we generate the elements of the sinusoidal encoding using the formulae:
$$E_{2i}=\sin(\frac{p}{10000^{\frac{2i}{128}}})$$
$$E_{2i+1}=\cos(\frac{p}{10000^{\frac{2i}{128}}})$$

- Here, $p\equiv$ position of the token, $i\equiv$ index of the element within the token

## Principle
- I understand this by considering a new matrix $864\times64$, where the 64 represents the grouped index of the sinusoidal encoding matrix. Consider each of these 64 items to be dials. Each dial represents a unit circle whose coordinates are specified by the member encodings.

- For example a value can be $(0.7,0.6)$. This coordinate represents a point in a dial (the circle) as all even indices are derived from a sine and odd indices from cosine $(\sin(x), \cos(x))$, the parametric form of a point in a circle

- For inital paired indices starting from 0, the coordinates will simply be $(\sin(p), \cos(p))$

- But we also not that, for tokens $15$ and $725$, their sine and cosine values are very close 


$$∣\sin(15)−\sin(725)∣≈4.58×10^{−5}$$

$$∣\cos(15)−\cos(725)∣≈3.92×10^{−5}$$

- This means they are extremely close as the sinusoidal frequency is comparitively high. This is where the other 64 dials come into picture. As we increase the index from 0 to 64 the power of the denominator rises exponentially and that leads to unique values that don't repeat for greater lengths 
- Combining all these 64 dials per token, we practically have a unique set of values that tells us the exact fingerprint of the token as well as the relative position of one toke with respect to another

- Note: The sinusoidal encoding has the useful mathematical property that a fixed positional offset corresponds to a linear transformation of the encoding. This motivated its use in the original Transformer. However, while this property makes relative position information readily available to the model, it does not prove that trained Transformers explicitly exploit this linear relationship. It may instead serve as a helpful inductive bias that makes learning positional relationships easier.