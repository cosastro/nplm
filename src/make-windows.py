import re
import collections
 
fileName = '../data/Fake/text.txt'
# fileName = 'Alkali metal.txt'
 
windowSize = 4
contextSize = 3
vocabulary = {}
 
with open(fileName) as file:
    buffer = file.read(windowSize)
    tail = ''
   
    while buffer != '':
        buffer = tail + buffer
        buffer = re.split('\.', buffer)
       
        tail = buffer[-1]
               
        for sentence in buffer[:-1]:
            words = re.split('\s+', sentence.strip())
           
            for wordIndex in range(len(words) - contextSize + 1):
                window = words[wordIndex: wordIndex + contextSize]
               
                print window
               
                if window[0] not in vocabulary:
                    vocabulary[window[0]] = 0
               
                vocabulary[window[0]] += 1
       
        words = re.split('\s+', tail.lstrip())

        buffer = file.read(windowSize)
       
        if len(words) > windowSize * 2 - 1 or buffer == '':
            if buffer != '':
                tail = ' '.join(words[-windowSize:])
                words = words[:-windowSize]
           
            for wordIndex in range(len(words) - contextSize + 1):
                window = words[wordIndex: wordIndex + contextSize]
               
                print window
               
                if window[0] not in vocabulary:
                    vocabulary[window[0]] = 0
               
                vocabulary[window[0]] += 1
       
#od = collections.OrderedDict(sorted(vocabulary.items(), key=lambda x: x[1]))
#for key, value in od.items():
#    print key, value