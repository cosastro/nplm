import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


floatX = theano.config.floatX

vocabularySize = 10
embeddingSize = 10
contextSize = 2
samples = 10
wordIndices = T.ivector('wordIndices')

defaultEmbeddings = np.arange(0, vocabularySize * embeddingSize).reshape((vocabularySize, embeddingSize)).astype(floatX)

embeddings = theano.shared(defaultEmbeddings, name='embeddings', borrow=True)

random = RandomStreams(seed=234)
negativeSampleIndices = random.random_integers((contextSize * samples,), 0, vocabularySize - 1)

indicies = T.concatenate([wordIndices, negativeSampleIndices])
indicies = indicies.reshape((samples + 1, contextSize))

output = embeddings[indicies]
output = output.mean(axis=1)

getEmbeddings = theano.function(
    inputs=[wordIndices],
    outputs=output
)

print getEmbeddings(range(0, contextSize))