import os
import random

pairs=set([('prova11','prova12',0),('prova21','prova22',1),('prova31','prova32',0)])

for element in pairs:
        print(element[0] + ' ' + element[1] + ' ' + str(element[2]) + '\n')