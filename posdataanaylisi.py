import csv
import pandas as pd
import matplotlib.pyplot as plt

corpora="corpus-kab.txt"
def  tagset (corpora):
    tags=[]
    for sentence in open(corpora,encoding='utf-8'):
        tagged_sentence=sentence.replace('\ufeff',"").replace('\n',"").split()
        for tagged_word in tagged_sentence:
            tag=tagged_word.split("/")[1]
            if tag not in tags:
                tags.append(tag)
    return tags

tags=tagset(corpora)


header = tags
data=[]

def initilialize (tags):
    words=[]
    for i in tags:
        words.append('')
    return words



for sentence in open(corpora,encoding='utf-8'):
        words=initilialize (tags)

        tagged_sentence=sentence.replace('\ufeff',"").replace('\n',"").split()
        for tagged_word in tagged_sentence:
            word=tagged_word.split("/")[0]
            words[tags.index(tagged_word.split("/")[1])]=word
        data.append(words)
#print (data)


with open('postag.csv', 'w', encoding='UTF8', newline='\n') as f:
    writer = csv.writer(f,delimiter='\t')

    # write the header
    writer.writerow(header)
    for i in data:
      writer.writerow(i)


df = pd.read_csv ('postag.csv',delimiter='\t')

Verbs=['VAF',   #aoriste futur
    'VAI',    # aoriste impératif
    'VAIT',   #aoriste intensif
    'VII',   #impératif intensif
    'VP',    # prétérit
    'VPA',   #participe aoriste
    'VPAIN', #participe aoriste intensif négatif
    'VPAIP', #participe aoriste intensif positif
    'VPN',   # prétérit négatif
    'VPPN',  #participe prétérit négatif
    'VPPP',  # participe prétérit positif
    'VS'     # verbe subjonctif
    ]
occurences=[]
for i in Verbs:

     occurences.append(df[i].count())

patches, texts, autotexts = plt.pie(occurences,
                                        labels=Verbs, autopct='%.0f%%',
                                        shadow=False, radius=1)
for t in texts:
        t.set_size('smaller')
autotexts[0].set_color('y')

plt.xlabel('Ifmiḍen n yimyagen s tmeẓri deg uḍris n ulmad')

plt.show()


##noms Verbes

Verbes=['VAF',   #aoriste futur
    'VAI',    # aoriste impératif
    'VAIT',   #aoriste intensif
    'VII',   #impératif intensif
    'VP',    # prétérit
    'VPA',   #participe aoriste
    'VPAIN', #participe aoriste intensif négatif
    'VPAIP', #participe aoriste intensif positif
    'VPN',   # prétérit négatif
    'VPPN',  #participe prétérit négatif
    'VPPP',  # participe prétérit positif
    'VS'     # verbe subjonctif
    ]

Verbs=['Imyagen','Ismawen']

occurences=[]
nb=0
for i in Verbes:
    nb=nb+df[i].count()


occurences.append(nb)

Names=['NMC',   #nom commun
       'NMP',    # nom propre
       'NCM',   #nom cardinal
]

nb=0
for i in Names:
    nb=nb+df[i].count()


occurences.append(nb)


patches, texts, autotexts = plt.pie(occurences,
                                        labels=Verbs, autopct='%.0f%%',
                                        shadow=False, radius=1)
for t in texts:
        t.set_size('smaller')
autotexts[0].set_color('y')

plt.xlabel('Ismawen d Imyagen')

plt.show()

