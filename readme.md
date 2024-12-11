# Q1.2 b)
    A meio das epocas a função exponencial está a dar crash o que me está a irritar levemente

# Q1.2 c)
    Os pdfs da norma são guardados automaticamente

# Q1.2 d)

    A utilização de l2 regularization vai fazer com que se tenha um modelo com uma distribuição de probabiliades mais smooth onde todos os Weights têm valores > 0, ou seja, todos os weights contribuem para o resultado final mesmo que uns menos que outros. Isto acontece na realidade porque a função de regularização é o somatório do quadrado dos pesos (O quadrado de um numero decimal é sempre um decimal menor logo com tendencia para 0)

    Ao usar l1 regularization, os pesos podem ter o valor de 0. Com isto as features menos importantes são ignoradas focando o modelo apenas nas features com importância para o que ele está a ser treinado.


Lembrei me quando estava a fazer o Q1.3 que não sei se tenho de por os bias no update weight, acredito que eles deviam ser postos logo na inicialização da matriz dos weights


# Q2.1 a)

    The learning rate that achieved the highst validation accuracy was the 0.001 with a validation accuracy of 0.5264.

    For the learning rate of 0.1, we have a highly unstable validation loss, flucutaitng throughout the trainning process. With this we can understand that the learning rate is too high and the model cannot adapt to the validation set.

    For the learning rate of 0.001 we reach a sweat spot. The validation loss decreases a lot in the beginning but plateus after the 15th epoch mantaining some minor fluctuations throught the remaining epochs. This shows us that the model is adapting well to the validation set which leads us to believe that it is learning well.

    For the learning rate of 0.00001, the validation loss decreases very slowly over the epochs with no sign of convergence. Since the difference between the train and validation loss is very small we can see that there is no major overfitting but at the same time, this small learning rate ist not efficient at improving the model.

# Q2.2 a)

- Default parameters - 5m 55s
    - Val acc 0.6033
    - Test acc 0.6013

- Batch 512 - 2m 5s
    - val acc 0.5306
    - test acc 0.5443

# Q2.2 b)

- Dropout 0.1
    - val acc 0.5798
    - test acc 0.5870

- Dropout 0.25
    - val acc 0.6033
    - test acc 0.6013

- Dropout 0.5
    - val acc 0.6111
    - test acc 0.6010


# Q2.2 c)

- Momentum 0.0
    - val acc 0.5014
    - test acc 0.5200

- Momentum 0.9
    - val acc 0.6019
    - test acc 0.6003