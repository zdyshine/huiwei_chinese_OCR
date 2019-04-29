# this version is too slow
# alphabets_file = open('./alphabets.py', 'w')
# txt_file = open('./train_data.txt', "r", encoding='utf-8')
# text_lines = txt_file.readlines()
# alphabets_file.write('alphabet = """')
# alphabets = ''
# text_list = ''
# for line in text_lines:
#     text = line.strip().split(' ')[1]
#     text_list += text
#     text_list = list(set(text_list))
#
# print(len(text_list))
# alphabets_file.write('"""')

import numpy as np
alphabet_list = []
train_word_file = './train_data.txt'

def readlines(train_word_file):
    with open(train_word_file, 'r') as f:
        return f.readlines()

def get_alphabet(train_word_file):
    lines =readlines(train_word_file)
    for line in lines:
        line = line.strip()
        image, sentence = line.strip().split('.jpg ')
        sentence = sentence.strip('"')
        for element in sentence:
            if element not in alphabet_list:
                alphabet_list.append(element)
    return alphabet_list


with open('./alphabets.py', 'w') as f0:
    alphabet = get_alphabet(train_word_file)
    print(len(alphabet))
    f0.write('alphabet = """')

    for element in alphabet:
        f0.write(str(element))
    f0.write('"""')
    f0.close()
