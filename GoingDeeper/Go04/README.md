# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 반태훈
- 리뷰어 : 이주연

# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - default box 로 얼굴4개를 모두 찾지는 못헀지만, 콧수염을 4명에게 모두 달았습니다 :)
      ```python
      def apply_sticker(img_raw, bbox, sticker_img):
        # 얼굴 바운딩 박스 좌표 (실제 이미지 크기 기준)
        img_h, img_w = img_raw.shape[:2]
        ymin, xmin, ymax, xmax = (np.array(bbox) * [img_h, img_w, img_h, img_w]).astype(int)
    
        face_width = xmax - xmin
        sticker_scale = 0.8  # 전체 박스의 80% 너비로
        sticker_width = int(face_width * sticker_scale)
        sticker_height = int(sticker_img.shape[0] * (sticker_width / sticker_img.shape[1]))
        resized_sticker = cv2.resize(sticker_img, (sticker_width, sticker_height))
    
        # 💡 얼굴 박스 하단 중심 기준으로 스티커 위치 설정
        x1 = int((xmin + xmax) / 2 - sticker_width / 2)
        y1 = int(ymax - sticker_height * 0.3)  # 살짝 위로 올려주기
    
        
    
        if resized_sticker.shape[2] == 4:
            sticker_rgb = resized_sticker[:, :, :3]
            mask = resized_sticker[:, :, 3] / 255.0
        else:
            sticker_rgb = resized_sticker
            mask = np.ones((sticker_height, sticker_width))
    
        h, w = sticker_rgb.shape[:2]
        roi = img_raw[y1:y1+h, x1:x1+w]
    
        if roi.shape[:2] == mask.shape:
            img_raw[y1:y1+h, x1:x1+w] = (roi * (1 - mask[..., np.newaxis]) + sticker_rgb * mask[..., np.newaxis]).astype(np.uint8)
    
        return img_raw
      ```
      ```python
        sticker_path = os.path.join(os.getenv("HOME"), "aiffel", "camera_sticker", "images", "cat-whiskers.png")
        sticker_img = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
        
        img_path = os.path.join(os.getenv("HOME"), "aiffel", "face_detector", "image_people.png")
        img_bgr = cv2.imread(img_path)
        
        # 스티커 적용
        result = apply_sticker_with_dlib(img_bgr, sticker_img)
        
        # 출력
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Sticker using dlib")
        plt.show()
      ```
    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    -네, 이해가 잘 되었습니다. 중간중간 설명이 잘 되어있습니다.
      ```python
        Step 2. SSD 모델을 통해 얼굴 bounding box 찾기
        잘 훈련된 모델을 통해 적절한 얼굴 bounding box를 찾아내기
        inference.py 코드 참고
        SSD의 Default box
      ```
        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 에포크를 60까지 돌려보는 시도를 했습니다. 하지만 얼굴을 찾는 정확도가 높아지지는 않았다고 했습니다.
      ```python
          EPOCHS = 60
      ```
        
- [ ]  **4. 회고를 잘 작성했나요?**
    - 네, 회고작성이 잘 되어있습니다. 러닝레이트에 대한 조정을 한번 더 시도했으면 좋았을 것이라는 아쉬움을 말씀해주셨습니다.
      ```python
      widerface 데이터셋이 사용되며, 경로가 PROJECT_PATH/widerface로 지정하고 wider_face_train_bbx_gt.txt를 파싱하여 bounding box 정보를 처리하는 parse_widerface()함수로 구현함(TFRecord 생성, augmentation, prior box 관련 코드도 구현함)
        SSD 관련 모듈 사용하고 bounding box 출력 및 시각화가 이루어짐
        에폭을 60까지만 돌리면 모델의 학습이 충분해 보인다는 팀원분의 말을듣고 진행하였으나, 여전히 3명만 face detection 하는 모습을 보여조금 아쉬웠음
        learnig rate를 조금 줄여서 학습할 껄 하는 아쉬움이 남음
        모델 최적화
        다양한 상황(테스트 상황)에 대한 분석
        특별한 상황(테스트 상황)에 대한 관찰
        테스트 데이터 모으기
        정량 지표 모으기
      ```
        
- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 네 간결하고 효율적입니다. dlib 사용할 때 필요한 코드만 가져왔습니다.
      ```python
          def apply_sticker_with_dlib(img_bgr, sticker_img):
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                faces = face_detector(img_rgb)
            
                for face in faces:
                    landmarks = landmark_model(img_rgb, face)
            
                    # 1. 기준점: 코 끝 아래쪽 (랜드마크 33)
                    nose_x = landmarks.part(30).x
                    # 코와 입 사이 중간 지점
                nose_y = int(landmarks.part(27).y)
        
        
                # 2. 스티커 크기 결정 (눈 사이 거리 기준)
                left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
                right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
                face_width = face.right() - face.left()
                
                sticker_width = int(face_width * 0.8)
                sticker_height = int(sticker_img.shape[0] * (sticker_width / sticker_img.shape[1]))
                resized_sticker = cv2.resize(sticker_img, (sticker_width, sticker_height))
        
                # 3. 스티커 위치 계산 (좌상단 기준)
                x1 = int(nose_x - sticker_width // 2)
                y1 = int(nose_y)
        
                # 4. 이미지 경계 체크
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_bgr.shape[1], x1 + sticker_width)
                y2 = min(img_bgr.shape[0], y1 + sticker_height)
        
                # 5. 알파 채널 분리
                if resized_sticker.shape[2] == 4:
                    sticker_rgb = resized_sticker[:, :, :3]
                    mask = resized_sticker[:, :, 3] / 255.0
                else:
                    sticker_rgb = resized_sticker
                    mask = np.ones((sticker_height, sticker_width))
        
                roi = img_bgr[y1:y2, x1:x2]
                h, w = roi.shape[:2]
                sticker_rgb = sticker_rgb[:h, :w]
                mask = mask[:h, :w]
        
                if roi.shape[:2] == mask.shape:
                    img_bgr[y1:y2, x1:x2] = (
                        roi * (1 - mask[..., np.newaxis]) + sticker_rgb * mask[..., np.newaxis]
                    ).astype(np.uint8)
        
            return img_bgr
      ```                                                        


# 회고(참고 링크 및 코드 개선)
```
필요한 부분만 코드를 잘 가져와 사용했습니다.
랜드마크를 적절히 사용해서 프로젝트 내용처럼 스티커를 잘 붙이는데 성공했습니다.
하지만 바운딩박스로 얼굴 4개를 찾아내는 부분은 3개만 찾게되어 아쉬움이 있습니다.
그래도 프로젝트 진행중에에 여러 고민과 시도를 하시느라 수고 많으셨습니다!


# 리뷰어의 회고를 작성합니다.
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
