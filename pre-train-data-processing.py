import json, re

wiAfile = open('./wi+locness/json/A.train.json')
train_data = open('train_file.txt', 'w')

for line in wiAfile:
    data = json.loads(line) #read in json data
    text = data['text']

    #remove leading and trailing \n and \t in order to allow newline to indicate document separation for ELECTRA, also remove multiple \n from within a text
    text = text.strip('\n')
    text = text.strip('\t')

    pattern  = re.compile('\n\n+')
    if pattern.search(text):
        text = pattern.sub('\n', text)
    #write to train_data file
    train_data.write(text)
    train_data.write('\n\n')


