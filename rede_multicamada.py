# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:50:59 2017

@author: André Louzada
"""

import numpy as np

def sigmoid(soma):
    #Função de ativação sigmoid
    return 1 / (1+np.exp(-soma))


def sigmoidDerivada(sig):
    #Calcula a derivada
    return sig*(1-sig)

entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])

saidas = np.array([[0],[1],[1],[0]])
 
#pesos0 = np.array([[-0.424, -0.740, -0.961],
 #                   [0.358, -0.577, -0.469]])

#Determina pesos aleatorios  
pesos0 = 2 * np.random.random((2,3)) - 1
  
#pesos1 = np.array([[-0.017],[-0.893], [0.148]])

#Determina o segundo peso aleatorio da camada
pesos1 = 2 * np.random.random((3,1)) - 1

epocas = 100000

#Taxa de aprendizagem da rede neural.
taxaDeAprendizagem = 0.6
momento = 1

for j in range(epocas):
    camadasEntradas = entradas
    somaSinapse0 = np.dot(camadasEntradas, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))

    print("A média do erro caiu para: " + str(mediaAbsoluta))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    peso1Transposta = pesos1.T
    
    deltaXPeso = deltaSaida.dot(peso1Transposta)
    
    deltaCamadaOculta = deltaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    
    #Gera um novo peso baseado na camada oculta
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaDeAprendizagem)
    
    camadaEntradaTransposta = camadasEntradas.T
    
    pesosNovos0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    
    pesos0 = (pesos0 * momento)+( pesosNovos0 * taxaDeAprendizagem)
    
    
    porcentagemAcerto = (1 - mediaAbsoluta) * 100

    print("------------------------------------")

    print("Acerto aumento em:" + str(porcentagemAcerto))
    
    