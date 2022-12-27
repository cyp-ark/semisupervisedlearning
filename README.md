# $\Pi$-Model from Scratch

## 1. Introduction
우리가 흔히 알고있는 머신러닝 방법인 지도학습과 비지도학습에 사용되는 데이터셋은 크게 두 종류로 나눌 수 있다. 지도학습은 정답을 학습해야 하기 때문에 라벨이 있는 데이터셋을 사용해야 하고, 비지도학습은 정답이 없는 데이터들을 이용해 패턴 등을 찾아내기 때문에 꼭 라벨이 있는 데이터를 사용하지 않아도 된다. 그렇다면 라벨이 있는 데이터와 라벨이 없는 데이터 중 어느 데이터셋이 실제로 더 많이 존재할까? 정답부터 말하자면 라벨이 없는 데이터가 훨씬 많은데, 그 이유는 라벨이 있는 데이터는 말 그대로 라벨을 데이터를 구축한 사람 또는 관련 분야의 전문가들이 직접 라벨링을 한 것이기 때문이다. 그렇기 때문에 우리는 전세계에서 만들어지는 데이터들 중 라벨링 된 극히 일부의 데이터셋만 이용해 모델을 훈련하고 검증하고 있는 것이다. 

이렇게 낭비되고 있는 라벨이 없는 데이터를 활용해 모델의 학습에 도움이 되도록 하는 방법을 준지도학습(Semi-Supervised Learning)이라고 한다. 준지도학습은 라벨이 없는 데이터도 활용해 모델의 학습에 도움을 주는데 이를 활용하기 위한 방벙들로는 라벨이 없는 데이터에 임의의 라벨을 달아주는 Pseudo-Labeling, 모델에서 라벨이 있는 데이터셋은 정답을 잘 맞추게 학습하고 라벨이 없는 데이터의 출력인 조금 변형을 주어도 그 결과값이 비슷해야 한다는 Consistency Regularization 방법 등이 있다.

그 중 우리는 Consistency Regularization 방법 중 비교적 초기에 제안된 모델인 $\Pi$-Model에 대해 알아보고 이를 실제 코드를 통해 구현해보는 시간을 가지고자 한다.


## 2. $\Pi$-Model
$\Pi$-Model은 Consistency Regularization 방법론이 가장 처음 제시된 Ladder Network와 $\Gamma$-Network 다음으로 제시된 모델이다. 기존의 두 모델은 두개의 인코더와 하나의 디코더를 사용해 layer에 noise가 적용된 corrupted encoder와 디코더를 통과한 hidden representation이 노이즈가 적용안된 clean encoder의 hidden representation이 같으면 좋은 모델일 것이다 라는 방법을 제안했다.
$\Pi$-Model은 굳이 hidden representation이 consistent할 필요가 없고, 두개의 Feed-Forward Network의 출력이 consistent한 모델이 더 직관적이고 좋을 것이다라고 제안한다. 

<p align="center"> <img src="https://github.com/cyp-ark/semisupervisedlearning/blob/main/figure/figure2.png?raw=true" width="60%" height="60%">

데이터의 입력단에 Gaussian noise를 통해 augmentation을 준 후 두개의 Network의 input으로 사용하는데 이때 두 Network는 Dropout을 다르게 주어 해당 입력이 다르게 학습되도록 한다. 각 Network의 출력을 각각 $z$와 $\tilde{z}$ 라고 할때 입력된 데이터가 라벨이 있는 labeled data의 경우 정답 $y$와 비교해 더 잘맞추는 방향으로 학습하고, 라벨이 없는 unlabeled data는 두 네트워크의 출력 $z$, $\tilde{z}$의 차이가 작아지는 방향으로 학습한다. Labeled data의 출력과 정답 $y$ 사이의 loss를 supervised loss라고 하고, unlabeled data의 출력 $z$, $\tilde{z}$ 사이의 loss를 unsupervised loss라고 한다. 해당 모델에서는 supervised loss로 Cross entropy를 사용하였고 unsupervised loss로는 Mean squared error(MSE)를 사용하였다.

<p align="center"> <img src="https://github.com/cyp-ark/semisupervisedlearning/blob/main/figure/figure1.png?raw=true" width="60%" height="60%">

## 3. $\Pi$-Model from Scratch
해당 모델을 python code를 이용해 직접 구현해보도록 하겠다. 모델의 학습에 이용한 데이터는 CIFAR-10이다.

<p align="center"> <img src="https://github.com/cyp-ark/semisupervisedlearning/blob/main/figure/figure3.png?raw=true" width="40%" height="40%">

CIFAR-10은 10개 범주로 구성된 32 $\times$ 32 이미지 데이터셋으로 train set 50000개, test set 10000개로 구성되어있다. CIFAR-10 데이터셋은 전부 라벨이 있기 때문에 모델을 학습하는 과정에서는 전체 데이터셋 90%의 라벨을 임의로 없애 실제 준지도학습을 적용해야하는 환경과 유사하게 설정하였다.

$\Pi$-Model의 Network architecture는 원 논문의 hyperparameter를 그대로 사용하였다.

<p align="center"> <img src="https://github.com/cyp-ark/semisupervisedlearning/blob/main/figure/figure4.png?raw=true" width="40%" height="40%">

해당 architecture를 keras를 이용하여 구현하면 다음과 같다.

```python
#Input 32x32 RGB image
img_input = keras.Input(shape=(32,32,3))
#GaussianNoise
aug_input = layers.GaussianNoise(stddev = 0.15)(img_input)

#conv1a
x1 = layers.Conv2D(128,(3,3),padding="same")(aug_input)
x1 = layers.LeakyReLU(alpha=0.1)(x1)
#conv1b
x1 = layers.Conv2D(128,(3,3),padding="same")(x1)
x1 = layers.LeakyReLU(alpha=0.1)(x1)
#conv1c
x1 = layers.Conv2D(128,(3,3),padding="same")(x1)
x1 = layers.LeakyReLU(alpha=0.1)(x1)

#pool1
x1 = layers.MaxPool2D((2,2))(x1)
#drop1
x1 = layers.Dropout(rate=0.5,seed = 13)(x1)

#conv2a
x1 = layers.Conv2D(256,(3,3),padding="same")(x1)
x1 = layers.LeakyReLU(alpha=0.1)(x1)
#conv2b
x1 = layers.Conv2D(256,(3,3),padding="same")(x1)
x1 = layers.LeakyReLU(alpha=0.1)(x1)
#conv2c
x1 = layers.Conv2D(256,(3,3),padding="same")(x1)
x1 = layers.LeakyReLU(alpha=0.1)(x1)

#pool2
x1 = layers.MaxPool2D((2,2))(x1)
#drop2
x1 = layers.Dropout(rate=0.5,seed = 57)(x1)

#conv3a
x1 = layers.Conv2D(512,(3,3),padding="valid")(x1)
x1 = layers.LeakyReLU(alpha=0.1)(x1)
#conv3b
x1 = layers.Conv2D(256,(1,1))(x1)
x1 = layers.LeakyReLU(alpha=0.1)(x1)
#conv3c
x1 = layers.Conv2D(128,(1,1))(x1)
x1 = layers.LeakyReLU(alpha=0.1)(x1)

#pool3
x1 = layers.GlobalAveragePooling2D()(x1)
#dense & output
output1 = layers.Dense(10, activation = "Softmax",name="z")(x1)
```

다른 하나의 network도 같게 구성해주면 되지만 추가적으로 unsupervised loss를 계산하기위한 layer를 추가해준다. 
```python
output2 = layers.subtract([x2,output1],name="z_tilde")
```

다음 코드를 통해 model을 구축한다. summary() 함수를 통해 해당 모델의 구성요소들을 알 수 있고, koras.utils.polt_model() 함수를 통해 모델의 구조를 시각적으로 표현 할 수 있다.
```python
model = keras.Model(inputs=img_input, outputs = [output1, output2])
model.summary()
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
```





## 4. Conclusion
