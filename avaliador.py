import pandas as pd
import cv2
import csv
from os import listdir

df = pd.read_csv("dataset.csv", sep=",", header=0)

pasta_modelos = "modelos"
modelos = listdir(pasta_modelos)

with open("avaliacao.csv", mode="w") as csv_file:
    fieldNames = ["modelo", "w_h", "num_stages","feature_type","tp", "fp", "fn", "tn", "precisao", "revocacao", "acuracia", "especificidade", "f_score"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldNames)
    writer.writeheader()

    for m in modelos:
        print("---------------------------------------------------------------")
        Tp, Tn, Fp, Fn = 0, 0, 0, 0

        print("Carregando modelo",m)
        minSize = int(m.split("_")[1])
        numStages = int(m.split("_")[2])
        featureType = m.split("_")[3]
        featureType = featureType.split(".")[0]

        cascadeModel = cv2.CascadeClassifier(pasta_modelos+"/"+m)

        for idx, i in df.iterrows():
            imgOri = cv2.imread("base_testes/"+i['imagem']+".png")
            imagem = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY)

            pistas = cascadeModel.detectMultiScale(imagem, 1.1, 5, 1, (minSize, minSize))

            if(len(pistas) > 0) and (i['contem_pista'] == 1):
                Tp = Tp + 1

            if(len(pistas) > 0) and (i['contem_pista'] == 0):
                Fp = Fp + 1

            if(len(pistas) == 0) and (i['contem_pista'] == 1):
                Fn = Fn + 1
            
            if(len(pistas) == 0) and (i['contem_pista'] == 0):
                Tn = Tn + 1

        precisao = Tp/(Tp + Fp)
        revocacao = Tp/(Tp + Fn)
        acuracia = (Tp + Tn)/(Tp + Tn + Fp + Fn)
        especificidade = Tn/(Tn+Fp)
        fScore = 2 * ((precisao * revocacao) / (precisao + revocacao))

        print("Matriz de confusão")
        print("Tp:", Tp)
        print("Fp:", Fp)
        print("Fn:", Fn)
        print("Tn:", Tn)
        print("Precisao:",precisao)
        print("Revocação:",revocacao)
        print("Acuracia:",acuracia)
        print("Especificidade:",especificidade)
        print("F-score:",fScore)

        writer.writerow({'modelo':m, 'w_h':str(minSize), 'num_stages':str(numStages), 'feature_type':featureType, 
        'tp':str(Tp), 'fp': str(Fp), 'fn':str(Fn), 'tn': str(Tn), 'precisao': str(precisao), 'revocacao':str(revocacao),
         'acuracia':str(acuracia), 'especificidade':str(especificidade), 'f_score':str(fScore)})