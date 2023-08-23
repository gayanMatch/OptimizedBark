import regex
import os
import random

CONVERSATION_DIR = '../Thomas and Cooper Following Up'

sentences = []
for filename in os.listdir(CONVERSATION_DIR):
    file = open(f"{CONVERSATION_DIR}/{filename}")
    data = file.read()
    file.close()
    matches = regex.findall(r":\n(.*)\n\n", data)
    for match in matches:
        word_num = match.count(' ')
        if word_num > 25:
            for sentence in match.split('.'):
                w_num = sentence.count(' ')
                if w_num > 5 and w_num < 25:
                    sentences.append(sentence)
        elif word_num < 5:
            pass
        else:
            sentences.append(match)

random.shuffle(sentences)
train_length = len(sentences) * 0.8
train_writer = open('train.txt', 'wt')
val_writer = open('val.txt', 'wt')
for i in range(len(sentences)):
    if i < train_length:
        train_writer.write(f"{sentences[i].strip()}\n")
    else:
        val_writer.write(f"{sentences[i].strip()}\n")
train_writer.close()
val_writer.close()
