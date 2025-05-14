# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 반태훈
- 리뷰어 : 양지웅


# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**  
    no_aug, aug 모델까지는 학습을 잘 진행하였으나, cutmix, mixup부터는 GPU문제로 제대로 완료하지 못하였다.
    ![image](https://github.com/user-attachments/assets/7d61e094-432d-4e41-b9fd-0d40803cb26f)  
    ![image](https://github.com/user-attachments/assets/353af511-c715-4763-970c-23d21012b46e)

    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**  
    모델을 구성하는 부분에서 세부 내용을 주석으로 잘 작성하였다.  
     ![image](https://github.com/user-attachments/assets/56f35db8-275a-4b06-968b-4af91526db7b)  

        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**  
    학습이 끊기는 경우를 대비하여 모델 체크포인트를 사용할 수 있다는 점을 기록해주었다.

    ![image](https://github.com/user-attachments/assets/c49a16c3-4ffd-4b49-b683-c83c9c4cf3f2)

        
- [x]  **4. 회고를 잘 작성했나요?**
    회고를 통해 아쉬웠던 점을 잘 기록해주었다. 이후 문제의 해결방안도 적어주었다.  
    ![image](https://github.com/user-attachments/assets/9d0ec270-fe87-4ea3-b60e-00442e0282c5)  

        
- [x]  **5. 코드가 간결하고 효율적인가요?**
      전처리 함수를 하나로 작성하여 이후 전처리에 쉽게 사용하였다.
     ![image](https://github.com/user-attachments/assets/42672cbe-440d-4fb1-a318-ebbf2879f74b)

    


# 회고(참고 링크 및 코드 개선)
```
서로의 코드와 실험 결과를 보고 피드백을 할 수 있었다. 데이터 증강이 일반화 성능에 미치는 영향을 파악하기 좋은 시간이었다.
```
