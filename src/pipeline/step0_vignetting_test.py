import cv2
import numpy as np
import os

class IntegratedCameraTester:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
        
        self.h, self.w = self.img.shape[:2]
        self.window_name = "Integrated Camera Tester"
        
        # 비네팅 보정용 거리 맵 미리 계산
        self.r2, self.r4 = self._precompute_dist_maps()
        
        # 초기 파라미터 설정
        self.cx, self.cy = self.w // 2, self.h // 2
        self.r = min(self.w, self.h) // 2
        self.k1, self.k2 = 0.0, 0.0
        self.cut_y = self.h

    def _precompute_dist_maps(self):
        X, Y = np.meshgrid(np.arange(self.w), np.arange(self.h))
        center_x, center_y = self.w / 2, self.h / 2
        dist_sq = ((X - center_x)**2 + (Y - center_y)**2) / (center_x**2 + center_y**2)
        return dist_sq, dist_sq**2

    def update(self, val):
        """트랙바 조절 시 호출되는 통합 업데이트 함수"""
        # 1. 현재 트랙바 값들 읽기
        self.cx = cv2.getTrackbarPos("Center X", self.window_name)
        self.cy = cv2.getTrackbarPos("Center Y", self.window_name)
        self.r = cv2.getTrackbarPos("Radius", self.window_name)
        self.k1 = cv2.getTrackbarPos("Vignette K1", self.window_name) / 100.0
        self.k2 = cv2.getTrackbarPos("Vignette K2", self.window_name) / 100.0
        self.cut_y = cv2.getTrackbarPos("Vehicle Cut Y", self.window_name)

        # 2. 비네팅 보정 적용
        vignette_gain = 1 + (self.k1 * self.r2) + (self.k2 * self.r4)
        corrected = self.img.astype(np.float32)
        for i in range(3):
            corrected[:, :, i] *= vignette_gain
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        # 3. 원형 마스크 생성 및 적용
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.circle(mask, (self.cx, self.cy), self.r, 255, -1)
        
        # 4. 차량 컷오프(하단) 적용
        cv2.rectangle(mask, (0, self.cut_y), (self.w, self.h), 0, -1)
        
        # 마스크 적용
        final_result = cv2.bitwise_and(corrected, corrected, mask=mask)

        # 5. 가이드라인 표시용 (원본 위에)
        display_origin = self.img.copy()
        cv2.circle(display_origin, (self.cx, self.cy), self.r, (0, 255, 0), 2) # 원형 가이드
        cv2.line(display_origin, (0, self.cut_y), (self.w, self.cut_y), (0, 0, 255), 2) # 차량 컷오프 가이드

        # 두 이미지를 합쳐서 표시
        combined = np.hstack((display_origin, final_result))
        cv2.imshow(self.window_name, combined)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 트랙바 생성
        cv2.createTrackbar("Center X", self.window_name, self.cx, self.w, self.update)
        cv2.createTrackbar("Center Y", self.window_name, self.cy, self.h, self.update)
        cv2.createTrackbar("Radius", self.window_name, self.r, max(self.w, self.h), self.update)
        cv2.createTrackbar("Vignette K1", self.window_name, 0, 200, self.update)
        cv2.createTrackbar("Vignette K2", self.window_name, 0, 200, self.update)
        cv2.createTrackbar("Vehicle Cut Y", self.window_name, self.h, self.h, self.update)
        
        self.update(0) # 초기 화면 로드
        
        print("--- 통합 테스트 시작 ---")
        print("'s' 키: 파라미터 출력 | 'q' 키: 종료")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('s'):
                print(f"\n[최종 파라미터]")
                print(f"MASK_CX: {self.cx}, MASK_CY: {self.cy}, MASK_R: {self.r}")
                print(f"VIGNETTE_K1: {self.k1:.2f}, VIGNETTE_K2: {self.k2:.2f}")
                print(f"VEHICLE_CUTOFF_Y: {self.cut_y}")
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    import glob
    from src.common import load_config

    parser = argparse.ArgumentParser(description="Step0: Camera vignetting tester")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--image", default=None, help="테스트할 이미지 경로 (없으면 projection.input_folder/blackfly의 첫 png)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.image:
        test_image_path = args.image
    else:
        candidates = sorted(glob.glob(os.path.join(cfg.projection.input_folder, "blackfly", "*.png")))
        if not candidates:
            raise FileNotFoundError(f"이미지를 찾을 수 없음: {cfg.projection.input_folder}/blackfly")
        test_image_path = candidates[0]

    if os.path.exists(test_image_path):
        tester = IntegratedCameraTester(test_image_path)
        tester.run()
    else:
        print(f"이미지 경로를 확인해주세요: {test_image_path}")