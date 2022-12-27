# $\Pi$-Model from Scratch

## 1. Introduction
우리가 흔히 알고있는 머신러닝 방법인 지도학습과 비지도학습에 사용되는 데이터셋은 크게 두 종류로 나눌 수 있다. 지도학습은 정답을 학습해야 하기 때문에 라벨이 있는 데이터셋을 사용해야 하고, 비지도학습은 정답이 없는 데이터들을 이용해 패턴 등을 찾아내기 때문에 꼭 라벨이 있는 데이터를 사용하지 않아도 된다. 그렇다면 라벨이 있는 데이터와 라벨이 없는 데이터 중 어느 데이터셋이 실제로 더 많이 존재할까? 정답부터 말하자면 라벨이 없는 데이터가 훨씬 많은데, 그 이유는 라벨이 있는 데이터는 말 그대로 라벨을 데이터를 구축한 사람 또는 관련 분야의 전문가들이 직접 라벨링을 한 것이기 때문이다. 그렇기 때문에 우리는 전세계에서 만들어지는 데이터들 중 라벨링 된 극히 일부의 데이터셋만 이용해 모델을 훈련하고 검증하고 있는 것이다. 

이렇게 낭비되고 있는 라벨이 없는 데이터를 활용해 모델의 학습에 도움이 되도록 하는 방법을 준지도학습(Semi-Supervised Learning)이라고 한다. 준지도학습은 라벨이 없는 데이터도 활용해 모델의 학습에 도움을 주는데 이를 활용하기 위한 방벙들로는 라벨이 없는 데이터에 임의의 라벨을 달아주는 Pseudo-Labeling, 모델에서 라벨이 있는 데이터셋은 정답을 잘 맞추게 학습하고 라벨이 없는 데이터의 출력인 조금 변형을 주어도 그 결과값이 비슷해야 한다는 Consistency Regularization 방법 등이 있다.

그 중 우리는 Consistency Regularization 방법 중 비교적 초기에 제안된 모델인 $\Pi$-Model에 대해 알아보고 이를 실제 코드를 통해 구현해보는 시간을 가지고자 한다.


## 2. $\Pi$-Model
$\Pi$-Model은 Consistency Regularization 방법론이 가장 처음 제시된 Ladder Network와 $\Gamma$-Network 다음으로 제시된 모델이다. 기존의 두 모델은 두개의 인코더와 하나의 디코더를 사용해 layer에 noise가 적용된 corrupted encoder와 디코더를 통과한 hidden representation이 노이즈가 적용안된 clean encoder의 hidden representation이 같으면 좋은 모델일 것이다 라는 방법을 제안했다. 


## 3. $\Pi$-Model from Scratch




## 4. Conclusion
