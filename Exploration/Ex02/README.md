# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 반태훈
- 리뷰어 : 이주연


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
    - 근거코드 
    ```python
    # 최종 모델 전체 학습(RMSLE=>RMSE)
    best_model.fit(X, y)
    test_pred = np.expm1(best_model.predict(X_test))
    ```

    
    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
     - 근거코드 
    ```python
    # 리노베이션 여부 및 주택 경과연수(외부 데이터를 쓸수 없으니 최대한 보기좋게 가공)
    # 리노베이션 여부를 0 또는 1로 표시
    # yr_renovated == 0 → 리노베이션 안 함 → 0
    # yr_renovated > 0 → 리노베이션 했음 → 1
    # "리노베이션 된 집이 더 비싼가?" 같은 걸 학습할 수 있다
    data['renovated'] = (data['yr_renovated'] != 0).astype(int)
    # 지어진 연도 그대로 쓰는 것보다, 이렇게 나이로 바꾸는 게 모델이 이해하기 쉬움
    data['age'] = data['yr_built'].max() - data['yr_built']
    # 이제 정보를 renovated랑 age로 요약했으니, 원본 컬럼은 제거
    data.drop(columns=['yr_renovated', 'yr_built'], inplace=True)

    ```

        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
    - 근거코드 
    ```python
    # 모델 + 파라미터
    models = {
        "RandomForest": (RandomForestRegressor(random_state=42), {
            'n_estimators': [100], 'max_depth': [10, None]
            # 실험할 하이퍼파라미터 조합: 2
        }),
        "XGBoost": (XGBRegressor(objective='reg:squarederror', random_state=42), {
            'n_estimators': [100], 'max_depth': [3, 6], 'learning_rate': [0.05, 0.1]
            # 실험할 하이퍼파라미터 조합: 4
        }),
        "LightGBM": (LGBMRegressor(random_state=42), {
            'n_estimators': [100], 'num_leaves': [31, 50], 'learning_rate': [0.05, 0.1]
            # 실험할 하이퍼파라미터 조합: 4
        })
    }
     ```
        
- [X]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
   - 근거코드 
    ```python
    #주요 구성

    #EDA 시각화 포함: missingno, seaborn 사용

    #피처 엔지니어링: 로그변환, 날짜 처리, 리노베이션 여부, 경과연수 등

    #모델 3종: RandomForest, XGBoost, LightGBM 전부 학습 + GridSearchCV 튜닝

    #성능 평가: RMSLE 기준으로 비교, 가장 좋은 모델 자동 선택

    #제출 파일 생성: submission_{모델명}_RMSLE_{점수}.csv 형태로 저장
    ``` 
        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
    - 근거코드 
    ```python
    # 모델 + 파라미터
    models = {
        "RandomForest": (RandomForestRegressor(random_state=42), {
            'n_estimators': [100], 'max_depth': [10, None]
            # 실험할 하이퍼파라미터 조합: 2
        }),
        "XGBoost": (XGBRegressor(objective='reg:squarederror', random_state=42), {
            'n_estimators': [100], 'max_depth': [3, 6], 'learning_rate': [0.05, 0.1]
            # 실험할 하이퍼파라미터 조합: 4
        }),
        "LightGBM": (LGBMRegressor(random_state=42), {
            'n_estimators': [100], 'num_leaves': [31, 50], 'learning_rate': [0.05, 0.1]
            # 실험할 하이퍼파라미터 조합: 4
        })
    }
     ```

# 회고(참고 링크 및 코드 개선)
```
# 리뷰어의 회고를 작성합니다.
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
