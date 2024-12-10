# Q1.2 b)
    A meio das epocas a função exponencial está a dar crash o que me está a irritar levemente

# Q1.2 c)
    Os pdfs da norma são guardados automaticamente

# Q1.2 d)

    A utilização de l2 regularization vai fazer com que se tenha um modelo com uma distribuição de probabiliades mais smooth onde todos os Weights têm valores > 0, ou seja, todos os weights contribuem para o resultado final mesmo que uns menos que outros. Isto acontece na realidade porque a função de regularização é o somatório do quadrado dos pesos (O quadrado de um numero decimal é sempre um decimal menor logo com tendencia para 0)

    Ao usar l1 regularization, os pesos podem ter o valor de 0. Com isto as features menos importantes são ignoradas focando o modelo apenas nas features com importância para o que ele está a ser treinado.