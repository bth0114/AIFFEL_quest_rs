# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 반태훈
- 리뷰어 : 이주연


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?** 
    - 문제를 해결한 이미지와 그래프가 잘 첨부되어 있습니다.
    - 
          ```python
            IoU 비교 결과 (U-Net vs U-Net++ vs U-Net++ with Deep Supervision)
                Index	U-Net	U-Net++	U-Net++ DS
                1	0.8495	0.8522	0.2813
                2	0.7254	0.6204	0.3218
                3	0.7435	0.6278	0.1977
                4	0.7571	0.5128	0.1873
                5	0.7319	0.5213	0.1756
                6	0.6685	0.4684	0.1359
                7	0.8664	0.7529	0.3127
                8	0.8139	0.7646	0.2625
                9	0.7769	0.7481	0.3442

        
          ```
          ```python
              #  시각화
                
                # 1. Loss
                plt.figure(figsize=(12, 6))
                plt.title("Losses per Output")
                
                plt.plot(history_U_net_plus_with_deep_suervision.history['loss'], label='Total Loss', color='black')
                plt.plot(history_U_net_plus_with_deep_suervision.history['conv2d_185_loss'], label='Loss (x04)', color='red')
                plt.plot(history_U_net_plus_with_deep_suervision.history['conv2d_184_loss'], label='Loss (x03)', color='orange')
                plt.plot(history_U_net_plus_with_deep_suervision.history['conv2d_183_loss'], label='Loss (x02)', color='green')
                plt.plot(history_U_net_plus_with_deep_suervision.history['conv2d_182_loss'], label='Loss (x01)', color='blue')
                
                plt.plot(history_U_net_plus_with_deep_suervision.history['val_loss'], label='Val Total Loss', color='black', linestyle='--')
                plt.plot(history_U_net_plus_with_deep_suervision.history['val_conv2d_185_loss'], label='Val Loss (x04)', color='red', linestyle='--')
                plt.plot(history_U_net_plus_with_deep_suervision.history['val_conv2d_184_loss'], label='Val Loss (x03)', color='orange', linestyle='--')
                plt.plot(history_U_net_plus_with_deep_suervision.history['val_conv2d_183_loss'], label='Val Loss (x02)', color='green', linestyle='--')
                plt.plot(history_U_net_plus_with_deep_suervision.history['val_conv2d_182_loss'], label='Val Loss (x01)', color='blue', linestyle='--')
                
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.show()
                
                # 2. Dice
                plt.figure(figsize=(12, 6))
                plt.title("Dice Coefficients per Output")
                
                plt.plot(history_U_net_plus_with_deep_suervision.history['conv2d_185_dice_coef'], label='Dice (x04)', color='red')
                plt.plot(history_U_net_plus_with_deep_suervision.history['conv2d_184_dice_coef'], label='Dice (x03)', color='orange')
                plt.plot(history_U_net_plus_with_deep_suervision.history['conv2d_183_dice_coef'], label='Dice (x02)', color='green')
                plt.plot(history_U_net_plus_with_deep_suervision.history['conv2d_182_dice_coef'], label='Dice (x01)', color='blue')
                
                plt.plot(history_U_net_plus_with_deep_suervision.history['val_conv2d_185_dice_coef'], label='Val Dice (x04)', color='red', linestyle='--')
                plt.plot(history_U_net_plus_with_deep_suervision.history['val_conv2d_184_dice_coef'], label='Val Dice (x03)', color='orange', linestyle='--')
                plt.plot(history_U_net_plus_with_deep_suervision.history['val_conv2d_183_dice_coef'], label='Val Dice (x02)', color='green', linestyle='--')
                plt.plot(history_U_net_plus_with_deep_suervision.history['val_conv2d_182_dice_coef'], label='Val Dice (x01)', color='blue', linestyle='--')
                
                plt.xlabel('Epoch')
                plt.ylabel('Dice Coefficient')
                plt.legend()
                plt.grid(True)
                plt.show()
          
          ```
    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 네, unet++ 구현시시 그림과 함께 코드가 작성되어 더 이해가 잘 되었습니다.
        
          ```python
            def build_U_net_model(input_shape=(224, 224, 3)):
            inputs = Input(shape=input_shape)
        
            # Encoder (Downsampling)
            c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
            c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
            p1 = MaxPooling2D((2, 2))(c1)
        
            c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
            c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
            p2 = MaxPooling2D((2, 2))(c2)
        
            c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
            c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
            p3 = MaxPooling2D((2, 2))(c3)
        
            c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
            c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
            p4 = MaxPooling2D((2, 2))(c4)
        
            # Bottleneck
            c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
            c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
        
            # Decoder (Upsampling)
            u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
            u6 = concatenate([u6, c4])
            c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
            c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
        
            u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
            u7 = concatenate([u7, c3])
            c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
            c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
        
            u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
            u8 = concatenate([u8, c2])
            c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
            c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
        
            u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
            u9 = concatenate([u9, c1])
            c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
            c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
        
            # Output layer
            outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        
            model = Model(inputs=[inputs], outputs=[outputs])
            return model
          ```
        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 네 Deep Supervision 에 대해 추가로 실험을 실행해보았습니다.
         
          ```python
              # U-Net++ with Deep Supervision 결과 (마지막 출력만 사용)
                input_image = imread(dir_path + image_file)
                processed = test_preproc(image=input_image)["image"]
                model_input = np.expand_dims(processed / 255.0, axis=0)
                
                output_ds = model_U_net_plus_with_deep_suervision.predict(model_input)
          
          ```
        
- [X]  **4. 회고를 잘 작성했나요?**
    - 네, 메트릭에 대한 서칭과 루브릭 기준으로 회고를 작성했습니다.
     
          ```python
                객체의 경계(Edge) 부분을 얼마나 잘 맞췄는지 평가할 수 있는 metric
                왜 일반 metric은 부족할까?
                일반적으로 segmentation에서는 이런 metric을 많이 씀:
                
                IoU (Intersection over Union)
                Dice coefficient
                Pixel accuracy
                이런 것들은 전체 픽셀 단위로 겹치는 정도를 평가해. 그런데 이 방식엔 단점 :
                
                경계(Edge)가 조금만 어긋나도 전체 IoU는 크게 떨어지지 않을 수 있음.
                
                그래서 경계 근처만 집중해서 평가하는 metric이 따로 필요
                
                이름	설명	장점	단점
                Boundary IoU (BIoU)	경계에서 일정 거리 이내만 따로 비교해서 IoU 계산	경계 예측 잘했는지 확인	구현 복잡도 있음
                Hausdorff Distance	예측 경계와 GT 경계 사이의 가장 먼 거리 측정	경계의 오차 민감하게 탐지	노이즈에 민감
                Boundary F1 Score (BF Score)	경계선에서 일정 거리 내 일치 여부로 F1 점수 계산	Precision/Recall 같이 비교	경계가 애매할 땐 애매하게 측정됨
                Chamfer Distance	경계 포인트들의 평균 거리 계산	매끄러운 경계 평가에 적합	실시간에 느림
                간단하게 경계 평가하고 싶다 → Boundary F1 Score
                정확한 거리 기반 경계 평가 원한다 → Hausdorff Distance
          ```
        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - Deep Supervision Outputs을 포함해서 unet++ 를 각 형식에 맞춰 가독성있게 코드를 작성하였습니다.
     
          ```python
            def build_U_net_plus_with_deep_suervision(input_shape=(224, 224, 3), num_classes=1, use_transpose=False, deep_supervision=False):
            inputs = Input(input_shape)
        
            # Encoder
            x00 = conv_block(inputs, 64)
            p0 = MaxPooling2D((2, 2))(x00)
        
            x10 = conv_block(p0, 128)
            p1 = MaxPooling2D((2, 2))(x10)
        
            x20 = conv_block(p1, 256)
            p2 = MaxPooling2D((2, 2))(x20)
        
            x30 = conv_block(p2, 512)
            p3 = MaxPooling2D((2, 2))(x30)
        
            x40 = conv_block(p3, 1024)
        
            # Decoder
            def up_block(x, filters):
                return Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x) if use_transpose else UpSampling2D((2, 2))(x)
        
            x01 = conv_block(concatenate([x00, up_block(x10, 64)]), 64)
            x11 = conv_block(concatenate([x10, up_block(x20, 128)]), 128)
            x21 = conv_block(concatenate([x20, up_block(x30, 256)]), 256)
            x31 = conv_block(concatenate([x30, up_block(x40, 512)]), 512)
        
            x02 = conv_block(concatenate([x00, x01, up_block(x11, 64)]), 64)
            x12 = conv_block(concatenate([x10, x11, up_block(x21, 128)]), 128)
            x22 = conv_block(concatenate([x20, x21, up_block(x31, 256)]), 256)
        
            x03 = conv_block(concatenate([x00, x01, x02, up_block(x12, 64)]), 64)
            x13 = conv_block(concatenate([x10, x11, x12, up_block(x22, 128)]), 128)
        
            x04 = conv_block(concatenate([x00, x01, x02, x03, up_block(x13, 64)]), 64)
        
            # Deep Supervision Outputs
            output_1 = Conv2D(num_classes, (1, 1), activation='sigmoid')(x01)
            output_2 = Conv2D(num_classes, (1, 1), activation='sigmoid')(x02)
            output_3 = Conv2D(num_classes, (1, 1), activation='sigmoid')(x03)
            output_4 = Conv2D(num_classes, (1, 1), activation='sigmoid')(x04)
        
            outputs = [output_1, output_2, output_3, output_4] if deep_supervision else [output_4]
        
            model = Model(inputs=inputs, outputs=outputs)
            return model
          ```


# 회고(참고 링크 및 코드 개선)
```
# Deep Supervision Output의 유무까지 결과치로 비교한 부분이 인상적이었습니다. 다음에 기회가 되면 조사한 메트릭도 같이 반영이 되면 더 풍성할 수 있을것 같습니다.
# loss 기준이 통일되지 않은 부분 (바이너리 크로스엔트로피, dice_loss) 을 통일화 해서 진행해보면 좋을 것 같다고 이야기 나눴습니다.
```
