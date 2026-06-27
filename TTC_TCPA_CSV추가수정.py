import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO


# ============================================================
# 1. 기본 설정
# ============================================================
CURRENT_DIR = Path(__file__).parent

# 실행할 영상 파일명만 바꾸세요.
VIDEO_PATH = CURRENT_DIR / "videos" / "2.mp4"


def select_run_mode():
    while True:
        selected = input(
            "실행 모드를 입력하세요 [normal=정상 scene 수집 / demo=AI 시연]: "
        ).strip().lower()
        if selected in {"normal", "demo"}:
            return selected
        print("normal 또는 demo 중 하나를 입력하세요.")


RUN_MODE = select_run_mode()

# YOLO 검출 모델
YOLO_MODEL_PATH = CURRENT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

# Observation/Risk 분리 Scene-based Transformer 모델
TRANSFORMER_RESULT_DIR = CURRENT_DIR / "transformer_results_obsrisk_scene_30frames"
TRANSFORMER_MODEL_PATH = TRANSFORMER_RESULT_DIR / "normal_obsrisk_scene_transformer.pt"
TRANSFORMER_METADATA_PATH = TRANSFORMER_RESULT_DIR / "normal_obsrisk_scene_transformer_metadata.json"
TRANSFORMER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 객체 검출 설정
DETECTION_CONF = 0.45
TRACKING_IOU = 0.50
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
PEDESTRIAN_CLASSES = {"person"}

# 정규화 BEV 설정
# 사용자가 찍는 6점 Homography Polygon 내부가 Risk Zone입니다.
VIDEO_NAME = VIDEO_PATH.stem
# 6점 Polygon Risk Zone 전용 설정파일: 기존 4점 설정과 분리하여 저장합니다.
CONFIG_FILE = CURRENT_DIR / f"{VIDEO_NAME}_normalized_bev_6point_polygon_config.json"
BEV_MAX_SIDE = 1000
DEFAULT_BEV_RATIO_WIDTH = 10.0
DEFAULT_BEV_RATIO_HEIGHT = 6.0
MIN_BEV_SIDE = 200

# 6점 Risk Zone 설정
RISK_POLYGON_POINT_COUNT = 6
# BEV 상에서 위쪽 폭을 조금 좁게 배치해 도로/횡단보도 형태를 표현합니다.
HEX_TOP_INSET_RATIO = 0.14
HEX_MIDDLE_HEIGHT_RATIO = 0.46

# Observation Zone 설정
# Risk Zone은 사용자가 지정한 6점 Polygon입니다.
# OBS는 해당 Polygon을 중심점 기준으로 주변까지 확장한 관찰영역입니다.
OBS_MARGIN = 0.15

# Scene-based Transformer 시퀀스 설정
SEQ_LEN = 30
INPUT_LEN = 20
PRED_LEN = 10
STRIDE = 10
MAX_FRAME_GAP = 2

# 시연 화면 구성
# False: 발표용 RULE 중심 화면 - AI NORMAL/ANOMALY 패널 및 실시간 Transformer 추론 숨김
# True : 필요 시 기존 Transformer 이상탐지 패널 재활성화
SHOW_AI_PANEL = False

# TTC-like + CPA 기반 RULE 위험도 설정
# CPA/T_CPA RULE은 6점 Risk Polygon 내부 객체를 중심으로 계산합니다.
# 시간 기반 위험도 기준
# 5초 이상: SAFE, 3~5초: CAUTION, 1~3초: WARNING, 0~1초: DANGER
PREDICTION_HORIZON_S = 5.0
SAFE_TIME_THRESHOLD_S = 5.0
CAUTION_TIME_THRESHOLD_S = 3.0
WARNING_TIME_THRESHOLD_S = 1.0
CPA_DISTANCE_THRESHOLD = 0.06
MIN_CLOSING_SPEED = 0.01
MIN_VEHICLE_SPEED_NORM_S = 0.02
CAUTION_RISK_THRESHOLD = 20
WARNING_RISK_THRESHOLD = 50
DANGER_RISK_THRESHOLD = 80

# 좌측 상단 위험도 디스플레이 평활화 설정
# 내부 RULE 위험도 계산은 변경하지 않고, 화면에 보이는 숫자와 상태만 자연스럽게 변화시킵니다.
DISPLAY_RISK_SMOOTHING_ENABLED = True
DISPLAY_RISK_RISE_STEP_PER_FRAME = 20.0   # 접근단계 점수를 빠르게 반영
DISPLAY_RISK_FALL_STEP_PER_FRAME = 8.0    # 순간 흔들림 시 급격한 하락 방지

# Observation Zone 접근 차량용 Forward Projection 위험도 설정
# 기존 CPA/T_CPA는 Risk Zone 내부 pair에 사용하고,
# 아래 점수는 Risk Zone 보행자 + Observation Zone 이동 차량의 사전 충돌경로 예측에 사용합니다.
ENABLE_FORWARD_PROJECTION_RULE = True
PROJECTION_HORIZON_S = 5.0
PROJECTION_STEPS = 30
PROJECTION_DISTANCE_THRESHOLD = 0.075

# 접근 중인 차량에 대한 단계적 위험도 표시 설정
# 기존 미래 위치 예측 결과(best_future_distance)는 그대로 사용하고,
# 위험거리 진입 전에도 접근 정도에 따라 낮은/중간 위험점수를 부여합니다.
ENABLE_PROGRESSIVE_APPROACH_RISK = True
APPROACH_MONITOR_MULTIPLIER = 3.20   # 위험거리의 3.2배 이내 예상 접근부터 점수 표시
APPROACH_CAUTION_MULTIPLIER = 1.70   # 위험거리의 1.7배 이내 예상 접근부터 CAUTION 가능
APPROACH_BASE_SCORE = 8.0            # 접근 경로가 관찰되기 시작할 때 초기 표시 점수
APPROACH_MONITOR_MAX_SCORE = 44.0    # 아직 가까운 접근이 아닌 경우 SAFE 범위 내 점수
APPROACH_CAUTION_MAX_SCORE = 69.0    # 위험거리 진입 전 조기 CAUTION 최대 점수

# 차량/보행자를 단일 점으로만 보지 않고, BEV에서 추정한 차체/신체 폭을
# Forward Projection 충돌 판정 거리 기준에 일부 반영합니다.
ENABLE_DYNAMIC_FOOTPRINT_THRESHOLD = True
VEHICLE_WIDTH_WEIGHT = 0.45
PERSON_WIDTH_WEIGHT = 0.20
COLLISION_MARGIN_NORM = 0.010
MAX_DYNAMIC_PROJECTION_THRESHOLD = 0.20
MIN_FOOTPRINT_WIDTH_NORM = 0.005
MAX_FOOTPRINT_WIDTH_NORM = 0.30

# RULE 전용 차량 이동 확정 필터
# Transformer feature의 원래 속도값은 변경하지 않고,
# CPA/T_CPA와 Forward Projection에 참여할 차량만 안정적으로 결정합니다.
RULE_MOTION_FILTER_ENABLED = True
RULE_SPEED_HISTORY_LEN = 8
RULE_MOVE_ENTER_WINDOW = 4
RULE_MOVE_ENTER_REQUIRED = 3
RULE_MOVE_ENTER_SPEED_NORM_S = 0.020
RULE_MOVE_ENTER_MEDIAN_NORM_S = 0.020
RULE_MOVE_EXIT_WINDOW = 6
RULE_MOVE_EXIT_REQUIRED = 5
RULE_MOVE_EXIT_SPEED_NORM_S = 0.010
RULE_MOVE_EXIT_MEDIAN_NORM_S = 0.012

# STOP 차량 occlusion/overlap guard:
# 이미 STOP 상태인 차량 박스가 사람/차량과 크게 겹치면
# bounding-box jitter에 의한 MOVING 승격을 일시 차단합니다.
STOP_OCCLUSION_GUARD_ENABLED = True
OVERLAP_SMALLER_BOX_RATIO_THRESHOLD = 0.22

PROJECTION_TRAJECTORY_THICKNESS = 6
PROJECTION_PAIR_LINE_THICKNESS = 6

# Transformer용 Scene feature
# RULE 결과값 자체는 입력하지 않습니다.
# Observation Zone에서 연속적으로 보이는 등장/접근 흐름과,
# Risk Zone 내부 상태를 함께 학습하도록 구성합니다.
FEATURE_NAMES = [
    "obs_person_count",
    "obs_vehicle_count",
    "obs_moving_vehicle_count",
    "risk_person_count",
    "risk_vehicle_count",
    "risk_moving_vehicle_count",
    "waiting_person_count",
    "approaching_vehicle_count",
    "obs_min_distance_norm",
    "obs_max_closing_speed_norm_s",
    "obs_min_cpa_distance_norm",
    "obs_min_time_to_cpa_s",
]
FEATURE_DIM = len(FEATURE_NAMES)
NO_PAIR_DISTANCE = 1.8  # Observation Zone 대각선보다 약간 큰 pair 없음 기본값

# 미니맵 시각화 설정: 화면 표시만 변경하며 학습/위험판정에는 영향 없음
MINIMAP_CANVAS_SIZE = 320
MINIMAP_POINT_RADIUS = 30          # BEV 원본상 점 크기: 축소 후에도 선명하게 보이도록 확대
MINIMAP_OUTLINE_RADIUS = 36
MINIMAP_OUTLINE_THICKNESS = 5
MINIMAP_LABEL_FONT_SCALE = 1.05
MINIMAP_LABEL_THICKNESS = 3
MINIMAP_RISK_LINE_THICKNESS = 7
MINIMAP_OPACITY = 0.92             # 1.0이면 완전 불투명, 낮을수록 원본 화면이 비침

# 미니맵 속도 시각화 설정: 실제 km/h가 아닌 정규화 BEV 속도(norm/s)
MINIMAP_SHOW_VELOCITY = True
MIN_ARROW_SPEED_NORM_S = 0.003
ARROW_MIN_LENGTH_PX = 75
ARROW_MAX_LENGTH_PX = 220
ARROW_SPEED_SCALE = 800.0
ARROW_THICKNESS = 8
ARROW_TIP_LENGTH = 0.28
SPEED_FONT_SCALE = 0.88
SPEED_TEXT_THICKNESS = 3

# 원본 영상 속도 시각화 설정: Risk Zone 내부 객체만 표시
# 실제 km/h가 아닌 정규화 BEV 속도(norm/s)를 사용합니다.
FRAME_SHOW_VELOCITY = True
FRAME_VECTOR_HORIZON_S = 0.60
FRAME_MIN_ARROW_SPEED_NORM_S = 0.003
FRAME_ARROW_MIN_LENGTH_PX = 35
FRAME_ARROW_MAX_LENGTH_PX = 125
FRAME_ARROW_SPEED_SCALE = 480.0
FRAME_ARROW_THICKNESS = 4
FRAME_ARROW_TIP_LENGTH = 0.25
FRAME_SPEED_FONT_SCALE = 0.55
FRAME_SPEED_TEXT_THICKNESS = 2

# Scene 데이터 저장 경로: 기존 scene/pair 방식과 분리
SCENE_CSV_DIR = CURRENT_DIR / "scene_features_obsrisk_30frames"
if RUN_MODE == "normal":
    DATA_LABEL = "normal"
    SEQUENCES_ROOT = CURRENT_DIR / "anomaly_sequences_normal_obsrisk_scene_30frames"
else:
    DATA_LABEL = "demo_anomaly_candidate"
    SEQUENCES_ROOT = CURRENT_DIR / "anomaly_sequences_demo_obsrisk_scene_30frames"

SCENE_CSV_FILE = SCENE_CSV_DIR / f"{VIDEO_NAME}_{RUN_MODE}_obsrisk_scene.csv"
VIDEO_SEQUENCE_DIR = SEQUENCES_ROOT / VIDEO_NAME
SEQUENCE_INDEX_FILE = VIDEO_SEQUENCE_DIR / "sequence_index.csv"

SCENE_CSV_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

# 같은 영상을 다시 처리할 때 해당 영상의 이전 시퀀스만 초기화
for old_file in VIDEO_SEQUENCE_DIR.glob("*.npy"):
    old_file.unlink()
if SEQUENCE_INDEX_FILE.exists():
    SEQUENCE_INDEX_FILE.unlink()


# ============================================================
# 2. 유틸리티 함수
# ============================================================
def calculate_bev_size(ratio_width, ratio_height):
    """실제 거리와 무관한 상대 비율로 정규화 BEV 출력 크기를 정합니다."""
    if ratio_width <= 0 or ratio_height <= 0:
        raise ValueError("가로/세로 비율은 0보다 커야 합니다.")

    if ratio_width >= ratio_height:
        bev_width = BEV_MAX_SIDE
        bev_height = int(round(BEV_MAX_SIDE * ratio_height / ratio_width))
    else:
        bev_height = BEV_MAX_SIDE
        bev_width = int(round(BEV_MAX_SIDE * ratio_width / ratio_height))

    return max(MIN_BEV_SIDE, bev_width), max(MIN_BEV_SIDE, bev_height)


def build_six_point_destination_polygon(bev_width, bev_height):
    """
    6개 원본 Risk 경계점의 BEV 대응점입니다.
    실제 Risk 판정에는 선택한 원본 6점이 Homography로 변환된 Polygon을 사용합니다.
    """
    top_inset = float(bev_width) * HEX_TOP_INSET_RATIO
    middle_y = float(bev_height) * HEX_MIDDLE_HEIGHT_RATIO
    return np.array([
        [top_inset, 0.0],
        [float(bev_width) - top_inset, 0.0],
        [float(bev_width), middle_y],
        [float(bev_width), float(bev_height)],
        [0.0, float(bev_height)],
        [0.0, middle_y],
    ], dtype=np.float32)


def get_homography_setup():
    """
    사용자가 클릭한 6점 Polygon을 Risk Zone으로 사용합니다.
    6개의 대응점을 이용해 하나의 평면 Homography를 계산하고,
    변환된 6점 Polygon 내부를 실제 위험판정 영역으로 사용합니다.
    """
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise RuntimeError(f"영상 첫 프레임을 읽지 못했습니다: {VIDEO_PATH}")

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"\n--- [{VIDEO_NAME}] 기존 6점 Polygon Risk Zone 설정이 있습니다 ---")
        print(
            f"기존 Risk BEV: {data.get('bev_width', '?')} x {data.get('bev_height', '?')} px "
            f"(비율 {data.get('ratio_width', '?')}:{data.get('ratio_height', '?')})"
        )
        ans_homo = input("6점 Risk Zone 설정을 새로 하시겠습니까? (y/n): ").strip().lower()
    else:
        print(f"\n--- [{VIDEO_NAME}] 6점 Polygon Risk Zone 설정 파일이 없습니다. 새로 설정합니다 ---")
        data = {"src_pts": [], "dst_pts": []}
        ans_homo = "y"

    invalid = (
        len(data.get("src_pts", [])) != RISK_POLYGON_POINT_COUNT
        or len(data.get("dst_pts", [])) != RISK_POLYGON_POINT_COUNT
        or data.get("coordinate_system") != "normalized_bev_6point_polygon"
    )

    if ans_homo == "y" or invalid:
        src_pts = []
        print("\n[6점 Risk Zone 설정] 실제 위험판정 대상인 도로/횡단보도 바닥 경계를 클릭하세요.")
        print("클릭 순서: 1 상단좌측 → 2 상단우측 → 3 우측중간 → 4 우측하단 → 5 좌측하단 → 6 좌측중간")
        print("주의: 6점은 사람/차량 위가 아니라 같은 도로 바닥 평면의 경계점으로 선택하세요.")
        print("선택한 6점 Polygon 내부가 곧 Risk Zone이 됩니다.")

        window_name = "Set 6-Point Risk Zone Polygon"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)

        def select_pts(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(src_pts) < RISK_POLYGON_POINT_COUNT:
                src_pts.append([x, y])

        cv2.setMouseCallback(window_name, select_pts)
        labels = ["1 TL", "2 TR", "3 RM", "4 BR", "5 BL", "6 LM"]

        while len(src_pts) < RISK_POLYGON_POINT_COUNT:
            temp = frame.copy()
            if len(src_pts) >= 2:
                cv2.polylines(
                    temp, [np.asarray(src_pts, dtype=np.int32)],
                    False, (0, 255, 255), 2, cv2.LINE_AA
                )
            for idx, point in enumerate(src_pts):
                cv2.circle(temp, tuple(point), 8, (0, 0, 255), -1)
                cv2.putText(
                    temp, labels[idx], (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2, cv2.LINE_AA
                )
            cv2.imshow(window_name, temp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow(window_name)
                raise KeyboardInterrupt("6점 Risk Zone 설정이 취소되었습니다.")

        preview = frame.copy()
        cv2.polylines(
            preview, [np.asarray(src_pts, dtype=np.int32)],
            True, (0, 255, 255), 3, cv2.LINE_AA
        )
        cv2.imshow(window_name, preview)
        cv2.waitKey(350)
        cv2.destroyWindow(window_name)
        data["src_pts"] = src_pts

        print("\n[정규화 Risk BEV 가로:세로 비율 설정]")
        print("실제 거리(m)가 아니라 조감도 화면의 상대적인 형태입니다.")
        try:
            width_input = input(f"가로 비율 [기본값 {DEFAULT_BEV_RATIO_WIDTH:g}]: ").strip()
            height_input = input(f"세로 비율 [기본값 {DEFAULT_BEV_RATIO_HEIGHT:g}]: ").strip()
            ratio_width = float(width_input) if width_input else DEFAULT_BEV_RATIO_WIDTH
            ratio_height = float(height_input) if height_input else DEFAULT_BEV_RATIO_HEIGHT
            bev_width, bev_height = calculate_bev_size(ratio_width, ratio_height)
        except ValueError:
            print("잘못된 입력입니다. 기본 비율 10:6을 적용합니다.")
            ratio_width = DEFAULT_BEV_RATIO_WIDTH
            ratio_height = DEFAULT_BEV_RATIO_HEIGHT
            bev_width, bev_height = calculate_bev_size(ratio_width, ratio_height)

        dst_pts = build_six_point_destination_polygon(bev_width, bev_height)
        data["dst_pts"] = dst_pts.tolist()
        data["coordinate_system"] = "normalized_bev_6point_polygon"
        data["ratio_width"] = ratio_width
        data["ratio_height"] = ratio_height
        data["bev_width"] = bev_width
        data["bev_height"] = bev_height

        matrix_check, _ = cv2.findHomography(
            np.float32(data["src_pts"]), np.float32(data["dst_pts"]), method=0
        )
        if matrix_check is None:
            raise RuntimeError("6점 Homography 계산에 실패했습니다. 점 순서와 위치를 다시 확인하세요.")

        transformed_risk_polygon = cv2.perspectiveTransform(
            np.array([data["src_pts"]], dtype=np.float32), matrix_check
        )[0]
        warped_view = cv2.warpPerspective(frame, matrix_check, (bev_width, bev_height))
        cv2.polylines(
            warped_view, [np.int32(np.round(transformed_risk_polygon))],
            True, (0, 255, 255), 4, cv2.LINE_AA
        )
        cv2.namedWindow("6-Point Risk BEV Check", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("6-Point Risk BEV Check", min(1200, bev_width), min(850, bev_height))
        cv2.imshow("6-Point Risk BEV Check", warped_view)
        print(f"생성된 6점 Risk BEV: {bev_width} x {bev_height} px")
        print(f"OBS: 6점 Risk Polygon 주변 {OBS_MARGIN * 100:.0f}% 확장영역")
        print("노란색 6점 경계가 의도한 위험영역인지 확인 후 아무 키나 누르세요.")
        cv2.waitKey(0)
        cv2.destroyWindow("6-Point Risk BEV Check")

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return (
        np.float32(data["src_pts"]),
        np.float32(data["dst_pts"]),
        int(data["bev_width"]),
        int(data["bev_height"]),
    )




def fit_minimap_to_square(minimap, canvas_size=300):
    """검은 BEV 미니맵 비율을 유지한 채 원본 화면 내부 패널에 배치합니다."""
    height, width = minimap.shape[:2]
    if width <= 0 or height <= 0:
        return np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    scale = min(canvas_size / width, canvas_size / height)
    display_w = max(1, int(round(width * scale)))
    display_h = max(1, int(round(height * scale)))
    resized = cv2.resize(minimap, (display_w, display_h))

    canvas = np.full((canvas_size, canvas_size, 3), 30, dtype=np.uint8)
    x0 = (canvas_size - display_w) // 2
    y0 = (canvas_size - display_h) // 2
    canvas[y0:y0 + display_h, x0:x0 + display_w] = resized
    cv2.rectangle(canvas, (0, 0), (canvas_size - 1, canvas_size - 1), (200, 200, 200), 2)
    return canvas


def draw_velocity_vector_on_minimap(minimap, record, color):
    """Risk Zone 내부 객체의 정규화 속도값만 표시합니다. 방향 화살표는 화면에 표시하지 않습니다."""
    if not MINIMAP_SHOW_VELOCITY:
        return

    start = np.asarray(record["pt_px"], dtype=np.float32)
    velocity = np.asarray(record.get("display_vel_norm_s", record["vel_norm_s"]), dtype=np.float32)
    speed = float(np.linalg.norm(velocity))

    text_x = int(np.clip(start[0] + MINIMAP_OUTLINE_RADIUS + 6, 5, minimap.shape[1] - 230))
    text_y = int(np.clip(start[1] + 48, 34, minimap.shape[0] - 10))
    speed_text = f"v={speed:.3f} norm/s"

    cv2.putText(
        minimap, speed_text, (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, SPEED_FONT_SCALE,
        (0, 0, 0), SPEED_TEXT_THICKNESS + 3, cv2.LINE_AA
    )
    cv2.putText(
        minimap, speed_text, (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, SPEED_FONT_SCALE,
        color if speed >= MIN_ARROW_SPEED_NORM_S else (190, 190, 190),
        SPEED_TEXT_THICKNESS, cv2.LINE_AA
    )




def draw_velocity_vector_on_frame(
    frame, record, box, color, inverse_homography_matrix, minimap_w, minimap_h
):
    """
    Risk Zone 내부 객체의 정규화 속도값만 원본 영상에 표시합니다.
    내부 위험판정에는 이동벡터를 계속 사용하지만, 방향 화살표는 화면에 표시하지 않습니다.
    """
    if not FRAME_SHOW_VELOCITY:
        return

    velocity_norm_s = np.asarray(record.get("display_vel_norm_s", record["vel_norm_s"]), dtype=np.float32)
    speed = float(np.linalg.norm(velocity_norm_s))
    box = np.asarray(box, dtype=int)

    text_x = int(np.clip(box[0], 5, frame.shape[1] - 195))
    text_y = int(np.clip(box[3] + 22, 22, frame.shape[0] - 8))
    motion_text = f"v={speed:.3f} norm/s"
    text_color = color if speed >= FRAME_MIN_ARROW_SPEED_NORM_S else (190, 190, 190)

    cv2.putText(
        frame, motion_text, (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, FRAME_SPEED_FONT_SCALE,
        (0, 0, 0), FRAME_SPEED_TEXT_THICKNESS + 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, motion_text, (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, FRAME_SPEED_FONT_SCALE,
        text_color, FRAME_SPEED_TEXT_THICKNESS, cv2.LINE_AA
    )




def make_kalman_filter(initial_point):
    """BEV 좌표에서 객체 위치와 속도를 평활화합니다. 기존 Kalman Filter를 유지합니다."""
    kf = cv2.KalmanFilter(4, 2, 0)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32
    )
    kf.statePost = np.array(
        [[initial_point[0]], [initial_point[1]], [0], [0]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def is_inside_polygon_norm(pt_norm, polygon_norm):
    """정규화 BEV 좌표의 점이 지정 Polygon 내부 또는 경계에 있는지 판정합니다."""
    point = (float(pt_norm[0]), float(pt_norm[1]))
    return cv2.pointPolygonTest(polygon_norm.astype(np.float32), point, False) >= 0


def is_inside_risk(pt_norm):
    """사용자가 지정한 6점 Polygon 내부를 TTC-like + CPA Risk Zone으로 사용합니다."""
    return is_inside_polygon_norm(pt_norm, RISK_POLYGON_NORM)


def is_inside_observation(pt_norm):
    """6점 Risk Polygon 주변으로 확장한 Observation Polygon 내부를 판단합니다."""
    return is_inside_polygon_norm(pt_norm, OBS_POLYGON_NORM)


def calculate_time_based_risk_percent(time_to_event_s):
    """
    TTC/T_CPA 시간 기준을 0~100 위험도 점수로 변환합니다.

    기준:
    - 5초 이상: SAFE
    - 3~5초 : CAUTION
    - 1~3초 : WARNING
    - 0~1초 : DANGER
    """
    if time_to_event_s is None or not np.isfinite(time_to_event_s):
        return 0

    t = float(np.clip(time_to_event_s, 0.0, SAFE_TIME_THRESHOLD_S))

    if t >= SAFE_TIME_THRESHOLD_S:
        return 0
    if t >= CAUTION_TIME_THRESHOLD_S:
        # 5 -> 20점, 3 -> 49점 근처
        return int(np.clip(
            CAUTION_RISK_THRESHOLD
            + (SAFE_TIME_THRESHOLD_S - t)
            / max(SAFE_TIME_THRESHOLD_S - CAUTION_TIME_THRESHOLD_S, 1e-6)
            * (WARNING_RISK_THRESHOLD - CAUTION_RISK_THRESHOLD - 1),
            CAUTION_RISK_THRESHOLD,
            WARNING_RISK_THRESHOLD - 1,
        ))
    if t >= WARNING_TIME_THRESHOLD_S:
        # 3 -> 50점, 1 -> 79점 근처
        return int(np.clip(
            WARNING_RISK_THRESHOLD
            + (CAUTION_TIME_THRESHOLD_S - t)
            / max(CAUTION_TIME_THRESHOLD_S - WARNING_TIME_THRESHOLD_S, 1e-6)
            * (DANGER_RISK_THRESHOLD - WARNING_RISK_THRESHOLD - 1),
            WARNING_RISK_THRESHOLD,
            DANGER_RISK_THRESHOLD - 1,
        ))

    # 1 -> 80점, 0 -> 100점
    return int(np.clip(
        DANGER_RISK_THRESHOLD
        + (WARNING_TIME_THRESHOLD_S - t)
        / max(WARNING_TIME_THRESHOLD_S, 1e-6)
        * (100 - DANGER_RISK_THRESHOLD),
        DANGER_RISK_THRESHOLD,
        100,
    ))


def update_display_risk_score(previous_score, raw_score):
    """
    내부 RULE 위험도는 그대로 두고, 좌측 상단 패널의 표시 점수만 부드럽게 변화시킵니다.
    상승은 빠르게, 하강은 천천히 반영하여 시연 중 숫자 튐을 완화합니다.
    """
    raw_score = float(np.clip(raw_score, 0.0, 100.0))
    previous_score = float(np.clip(previous_score, 0.0, 100.0))

    if not DISPLAY_RISK_SMOOTHING_ENABLED:
        return raw_score

    if raw_score > previous_score:
        return min(raw_score, previous_score + DISPLAY_RISK_RISE_STEP_PER_FRAME)
    if raw_score < previous_score:
        next_score = max(raw_score, previous_score - DISPLAY_RISK_FALL_STEP_PER_FRAME)
        return 0.0 if next_score < 0.5 else next_score
    return previous_score


def get_display_risk_state(display_score):
    """평활화된 표시 점수에 따라 좌측 상단의 상태/색상만 결정합니다."""
    if display_score >= DANGER_RISK_THRESHOLD:
        return "DANGER", (0, 0, 255)
    if display_score >= WARNING_RISK_THRESHOLD:
        return "WARNING", (0, 165, 255)
    if display_score >= CAUTION_RISK_THRESHOLD:
        return "CAUTION", (0, 255, 255)
    return "SAFE", (0, 255, 0)


def draw_rule_panel(frame, status, color, risk_percent, rule_source="NONE"):
    risk_percent = max(0, min(100, int(risk_percent)))
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (820, 108), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.rectangle(frame, (20, 20), (820, 108), (255, 255, 255), 1)
    text = f"RULE STATUS: {status} ({risk_percent}%)"
    source_text = f"SOURCE: {rule_source}"
    cv2.putText(frame, text, (40, 66), cv2.FONT_HERSHEY_DUPLEX, 1.00, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, (40, 66), cv2.FONT_HERSHEY_DUPLEX, 1.00, color, 2, cv2.LINE_AA)
    cv2.putText(frame, source_text, (42, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame




def draw_risk_basis_panel(frame, basis):
    """
    위험도 점수의 계산 근거를 화면에 표시합니다.
    교수님 질문 대비용: 현재거리, 접근속도, TTC, T_CPA, CPA distance를 직접 보여줍니다.
    단위는 실제 m, km/h가 아니라 BEV 정규화 좌표계 기준입니다.
    """
    x1, y1, x2, y2 = 20, 118, 940, 282
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    if basis is None:
        lines = [
            "Risk basis: no valid approaching vehicle-person pair",
            "D_now: -   Closing: -   TTC: -",
            "T_CPA: -   CPA_dist: -",
            "TTC = D_now / Closing,  T_CPA = -dot(r, v) / |v|^2",
        ]
        color = (210, 210, 210)
    else:
        lines = [
            f"Risk basis: {basis.get('source', 'RULE')}",
            f"D_now: {basis.get('distance', 0.0):.3f} norm   "
            f"Closing: {basis.get('closing_speed', 0.0):.3f} norm/s   "
            f"TTC: {basis.get('ttc', 0.0):.2f}s",
            f"T_CPA: {basis.get('t_cpa', 0.0):.2f}s   "
            f"CPA_dist: {basis.get('cpa_distance', 0.0):.3f} norm   "
            f"Limit: {basis.get('limit', 0.0):.3f}",
            "TTC = D_now / Closing,  T_CPA = -dot(r, v) / |v|^2",
        ]
        color = (255, 255, 255)

    for idx, line in enumerate(lines):
        y = y1 + 34 + idx * 34
        cv2.putText(
            frame, line, (x1 + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.68,
            (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            frame, line, (x1 + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.68,
            color, 1, cv2.LINE_AA
        )

    return frame

def draw_ai_panel(frame, ai_state, ai_color, ai_score, threshold, collected_frames):
    """SHOW_AI_PANEL=True일 때만 사용하는 선택적 Transformer 표시 패널입니다."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 118), (830, 206), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.rectangle(frame, (20, 118), (830, 206), (255, 255, 255), 1)

    if ai_score is None:
        text = f"AI: COLLECTING OBSERVATION ({collected_frames}/{SEQ_LEN})"
    else:
        text = f"AI: {ai_state}  MSE {ai_score:.3f} / THR {threshold:.3f}"

    cv2.putText(frame, text, (40, 171), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (40, 171), cv2.FONT_HERSHEY_DUPLEX, 0.72, ai_color, 2, cv2.LINE_AA)
    return frame


def draw_scene_summary(frame, scene_vector):
    text = (
        f"OBS: P={int(scene_vector[0])} V={int(scene_vector[1])} M={int(scene_vector[2])}  "
        f"RISK: P={int(scene_vector[3])} V={int(scene_vector[4])} M={int(scene_vector[5])}"
    )
    cv2.putText(frame, text, (30, frame.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (30, frame.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


# ============================================================
# 3. Observation Scene Feature 및 Risk Rule
# ============================================================
def calculate_overlap_over_smaller_box(box_a, box_b):
    """
    두 bounding box의 교집합을 작은 박스 면적으로 나눈 값입니다.
    큰 차량과 작은 보행자의 겹침은 IoU보다 이 지표가 더 민감합니다.
    """
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter_area / min(area_a, area_b)


def is_stopped_vehicle_occluded(
    vehicle_box, vehicle_track_id, all_boxes, all_track_ids, all_classes, class_names
):
    """
    현재 STOP 차량이 다른 검출 객체와 겹쳐 박스 흔들림이 발생할 수 있는지 확인합니다.
    이미 MOVING인 차량의 상태 유지는 이 함수로 차단하지 않습니다.
    """
    if not STOP_OCCLUSION_GUARD_ENABLED:
        return False

    for other_box, other_id, other_cls in zip(all_boxes, all_track_ids, all_classes):
        if int(other_id) == int(vehicle_track_id):
            continue

        other_name = class_names[int(other_cls)]
        if other_name not in VEHICLE_CLASSES and other_name not in PEDESTRIAN_CLASSES:
            continue

        overlap_ratio = calculate_overlap_over_smaller_box(vehicle_box, other_box)
        if overlap_ratio >= OVERLAP_SMALLER_BOX_RATIO_THRESHOLD:
            return True

    return False


def update_rule_vehicle_motion_state(vehicle_record, motion_states, block_stop_transition=False):
    """
    RULE 전용 이동상태 필터입니다.

    - Transformer용 원래 vel_norm_s는 변경하지 않습니다.
    - rule_vel_norm_s / rule_moving만 CPA/T_CPA 및 Forward Projection에 사용합니다.
    - STOP 차량이 다른 객체와 겹친 경우, 가림에 따른 box 흔들림으로 MOVING 승격되지 않도록 보호합니다.
    - 이미 MOVING인 차량은 겹침 상황에서도 위험 계산을 유지합니다.
    """
    raw_velocity = np.asarray(vehicle_record["vel_norm_s"], dtype=np.float32)
    raw_speed = float(np.linalg.norm(raw_velocity))
    track_id = int(vehicle_record["id"])

    if not RULE_MOTION_FILTER_ENABLED:
        vehicle_record["rule_moving"] = raw_speed > MIN_VEHICLE_SPEED_NORM_S
        vehicle_record["rule_vel_norm_s"] = raw_velocity.copy()
        vehicle_record["rule_speed_norm_s"] = raw_speed
        vehicle_record["display_vel_norm_s"] = raw_velocity.copy()
        vehicle_record["stop_occ_guarded"] = False
        return

    state = motion_states.get(track_id)
    if state is None:
        state = {
            "speed_history": deque(maxlen=RULE_SPEED_HISTORY_LEN),
            "velocity_history": deque(maxlen=RULE_SPEED_HISTORY_LEN),
            "moving": False,
        }
        motion_states[track_id] = state

    # 이미 이동 중인 차량은 사람과 겹쳐도 사고 접근을 놓치지 않도록 계속 추적합니다.
    # 반대로 STOP 차량이 겹침 상태이면, 해당 구간의 박스 이동은 신뢰하지 않습니다.
    guard_active = bool(block_stop_transition and not state["moving"])
    vehicle_record["stop_occ_guarded"] = guard_active

    if guard_active:
        state["speed_history"].append(0.0)
        state["velocity_history"].append(np.zeros(2, dtype=np.float32))
        rule_velocity = np.zeros(2, dtype=np.float32)

        vehicle_record["rule_moving"] = False
        vehicle_record["rule_vel_norm_s"] = rule_velocity
        vehicle_record["rule_speed_norm_s"] = 0.0
        vehicle_record["display_vel_norm_s"] = rule_velocity
        return

    state["speed_history"].append(raw_speed)
    state["velocity_history"].append(raw_velocity.copy())
    speeds = np.asarray(state["speed_history"], dtype=np.float32)

    if not state["moving"]:
        if len(speeds) >= RULE_MOVE_ENTER_WINDOW:
            recent = speeds[-RULE_MOVE_ENTER_WINDOW:]
            high_count = int(np.sum(recent >= RULE_MOVE_ENTER_SPEED_NORM_S))
            median_speed = float(np.median(recent))
            if (
                high_count >= RULE_MOVE_ENTER_REQUIRED
                and median_speed >= RULE_MOVE_ENTER_MEDIAN_NORM_S
            ):
                state["moving"] = True
    else:
        if len(speeds) >= RULE_MOVE_EXIT_WINDOW:
            recent = speeds[-RULE_MOVE_EXIT_WINDOW:]
            low_count = int(np.sum(recent <= RULE_MOVE_EXIT_SPEED_NORM_S))
            median_speed = float(np.median(recent))
            if (
                low_count >= RULE_MOVE_EXIT_REQUIRED
                and median_speed <= RULE_MOVE_EXIT_MEDIAN_NORM_S
            ):
                state["moving"] = False

    velocity_history = np.asarray(state["velocity_history"], dtype=np.float32)
    robust_velocity = np.median(velocity_history, axis=0).astype(np.float32)

    rule_velocity = robust_velocity if state["moving"] else np.zeros(2, dtype=np.float32)

    vehicle_record["rule_moving"] = bool(state["moving"])
    vehicle_record["rule_vel_norm_s"] = rule_velocity
    vehicle_record["rule_speed_norm_s"] = float(np.linalg.norm(rule_velocity))
    vehicle_record["display_vel_norm_s"] = rule_velocity




def calculate_bev_footprint_width_norm(box, homography_matrix, minimap_w, minimap_h):
    """
    원본 영상 바운딩박스의 바닥 좌우점을 BEV로 변환해 정규화 폭을 계산합니다.
    지면 접점에 가까운 하단 폭만 사용하여 정규화 BEV 기반 RULE과 단위를 맞춥니다.
    """
    x1, _, x2, y2 = [float(value) for value in box]
    bottom_corners = np.array([[[x1, y2], [x2, y2]]], dtype=np.float32)
    mapped_corners = cv2.perspectiveTransform(bottom_corners, homography_matrix)[0]

    left_norm = np.array([
        mapped_corners[0][0] / max(minimap_w, 1),
        mapped_corners[0][1] / max(minimap_h, 1),
    ], dtype=np.float32)
    right_norm = np.array([
        mapped_corners[1][0] / max(minimap_w, 1),
        mapped_corners[1][1] / max(minimap_h, 1),
    ], dtype=np.float32)

    width_norm = float(np.linalg.norm(right_norm - left_norm))
    return float(np.clip(width_norm, MIN_FOOTPRINT_WIDTH_NORM, MAX_FOOTPRINT_WIDTH_NORM))


def get_dynamic_projection_threshold(vehicle, person):
    """
    Forward Projection 전용 충돌거리 기준입니다.
    기본 고정 threshold보다 작아지지 않으며, 큰 차량/보행자에 대해서만
    차체 측면 근접을 반영할 수 있도록 제한적으로 증가합니다.
    """
    if not ENABLE_DYNAMIC_FOOTPRINT_THRESHOLD:
        return PROJECTION_DISTANCE_THRESHOLD

    vehicle_width = float(vehicle.get("width_norm", MIN_FOOTPRINT_WIDTH_NORM))
    person_width = float(person.get("width_norm", MIN_FOOTPRINT_WIDTH_NORM))

    footprint_threshold = (
        COLLISION_MARGIN_NORM
        + VEHICLE_WIDTH_WEIGHT * vehicle_width
        + PERSON_WIDTH_WEIGHT * person_width
    )
    return float(np.clip(
        max(PROJECTION_DISTANCE_THRESHOLD, footprint_threshold),
        PROJECTION_DISTANCE_THRESHOLD,
        MAX_DYNAMIC_PROJECTION_THRESHOLD,
    ))


def calculate_pair_metrics(vehicles, pedestrians):
    """
    지정된 목록의 vehicle-person pair에서 장면 요약값을 계산합니다.
    Transformer에는 Observation Zone pair의 연속 지표만 사용합니다.
    """
    min_distance = NO_PAIR_DISTANCE
    max_closing_speed = 0.0
    min_cpa_distance = NO_PAIR_DISTANCE
    min_time_to_cpa = PREDICTION_HORIZON_S

    for vehicle in vehicles:
        for person in pedestrians:
            relative_position = person["pt_norm"] - vehicle["pt_norm"]
            relative_velocity = person["vel_norm_s"] - vehicle["vel_norm_s"]
            distance = float(np.linalg.norm(relative_position))
            min_distance = min(min_distance, distance)

            closing_speed = float(
                -np.dot(relative_velocity, relative_position) / (distance + 1e-6)
            )
            max_closing_speed = max(max_closing_speed, max(0.0, closing_speed))

            relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
            if closing_speed > 1e-6 and relative_speed_sq > 1e-8:
                time_to_cpa = float(
                    -np.dot(relative_position, relative_velocity) / relative_speed_sq
                )
                if 0.0 < time_to_cpa < PREDICTION_HORIZON_S:
                    position_at_cpa = relative_position + relative_velocity * time_to_cpa
                    distance_at_cpa = float(np.linalg.norm(position_at_cpa))
                    min_cpa_distance = min(min_cpa_distance, distance_at_cpa)
                    min_time_to_cpa = min(min_time_to_cpa, time_to_cpa)

    return min_distance, max_closing_speed, min_cpa_distance, min_time_to_cpa


def calculate_rule_risk(risk_vehicles, risk_pedestrians, obs_vehicles, minimap):
    """
    최종 RULE 위험도:
    1) 기존 CPA/T_CPA: Risk Zone 내부 차량-보행자 pair의 실제 근접 위험 판단
    2) Forward Projection: Risk Zone 내부 보행자와 Observation Zone 접근 차량의
       미래 이동경로가 가까워질 때 사전 위험 판단

    Transformer 학습/추론 입력에는 영향을 주지 않으며 RULE 표시만 보완합니다.
    """
    max_rule_risk_percent = 0
    rule_source = "NONE"
    best_rule_line = None
    best_rule_text = None
    best_projection_paths = None
    best_risk_basis = None

    # --------------------------------------------------------
    # A. 기존 CPA / T_CPA 위험도: Risk Zone 내부 pair만
    # --------------------------------------------------------
    for vehicle in risk_vehicles:
        vehicle_velocity = np.asarray(
            vehicle.get("rule_vel_norm_s", vehicle["vel_norm_s"]), dtype=np.float32
        )
        vehicle_speed = float(np.linalg.norm(vehicle_velocity))
        if not vehicle.get("rule_moving", vehicle_speed > MIN_VEHICLE_SPEED_NORM_S):
            continue
        if vehicle_speed <= MIN_VEHICLE_SPEED_NORM_S:
            continue

        for person in risk_pedestrians:
            relative_position = person["pt_norm"] - vehicle["pt_norm"]
            relative_velocity = person["vel_norm_s"] - vehicle["vel_norm_s"]
            distance = float(np.linalg.norm(relative_position))
            closing_speed = float(
                -np.dot(relative_velocity, relative_position) / (distance + 1e-6)
            )
            relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))

            if closing_speed <= MIN_CLOSING_SPEED or relative_speed_sq <= 1e-8:
                continue

            # TTC: 현재 거리 / 접근속도. BEV 정규화 좌표계 기준의 충돌 예상 시간입니다.
            ttc = float(distance / (closing_speed + 1e-6))

            time_to_cpa = float(
                -np.dot(relative_position, relative_velocity) / relative_speed_sq
            )
            if not (0.0 < time_to_cpa < PREDICTION_HORIZON_S):
                continue

            position_at_cpa = relative_position + relative_velocity * time_to_cpa
            distance_at_cpa = float(np.linalg.norm(position_at_cpa))
            if distance_at_cpa >= CPA_DISTANCE_THRESHOLD:
                continue

            pair_risk_percent = calculate_time_based_risk_percent(time_to_cpa)
            if pair_risk_percent > max_rule_risk_percent:
                max_rule_risk_percent = pair_risk_percent
                rule_source = "CPA / T_CPA"
                best_projection_paths = None
                best_rule_line = (
                    (int(person["pt_px"][0]), int(person["pt_px"][1])),
                    (int(vehicle["pt_px"][0]), int(vehicle["pt_px"][1])),
                )
                best_rule_text = (
                    (max(0, int(vehicle["pt_px"][0])),
                     max(48, int(vehicle["pt_px"][1]) - 15)),
                    f"TTC:{ttc:.2f}s T_CPA:{time_to_cpa:.2f}s CPA:{distance_at_cpa:.3f}",
                )
                best_risk_basis = {
                    "source": "CPA / T_CPA",
                    "distance": distance,
                    "closing_speed": closing_speed,
                    "ttc": ttc,
                    "t_cpa": time_to_cpa,
                    "cpa_distance": distance_at_cpa,
                    "limit": CPA_DISTANCE_THRESHOLD,
                }

    # --------------------------------------------------------
    # B. 미래 위치 투영 위험도:
    #    보행자는 실제 횡단 위험영역(Risk Zone) 내부만,
    #    차량은 Observation Zone 안의 접근 차량까지 포함
    # --------------------------------------------------------
    if ENABLE_FORWARD_PROJECTION_RULE and risk_pedestrians:
        future_times = np.linspace(
            PROJECTION_HORIZON_S / PROJECTION_STEPS,
            PROJECTION_HORIZON_S,
            PROJECTION_STEPS,
            dtype=np.float32
        )

        for vehicle in obs_vehicles:
            vehicle_velocity = np.asarray(
                vehicle.get("rule_vel_norm_s", vehicle["vel_norm_s"]), dtype=np.float32
            )
            vehicle_speed = float(np.linalg.norm(vehicle_velocity))
            if not vehicle.get("rule_moving", vehicle_speed > MIN_VEHICLE_SPEED_NORM_S):
                continue
            if vehicle_speed <= MIN_VEHICLE_SPEED_NORM_S:
                continue

            for person in risk_pedestrians:
                relative_position = person["pt_norm"] - vehicle["pt_norm"]
                relative_velocity = person["vel_norm_s"] - vehicle_velocity
                initial_distance = float(np.linalg.norm(relative_position))
                closing_speed = float(
                    -np.dot(relative_velocity, relative_position) / (initial_distance + 1e-6)
                )

                # 현재부터 보행자 쪽으로 접근하는 차량만 사전예측 대상
                if closing_speed <= MIN_CLOSING_SPEED:
                    continue

                # TTC: 현재 거리 / 접근속도. Forward Projection 구간에서도 화면 표시용 근거값으로 사용합니다.
                ttc = float(initial_distance / (closing_speed + 1e-6))

                best_future_distance = float("inf")
                best_future_time = None
                best_person_future = None
                best_vehicle_future = None

                for future_time in future_times:
                    person_future = person["pt_norm"] + person["vel_norm_s"] * future_time
                    vehicle_future = vehicle["pt_norm"] + vehicle_velocity * future_time

                    # 보행자가 미래에도 실제 Risk 영역에 있는 경우만 경고 대상으로 유지
                    if not is_inside_risk(person_future):
                        continue

                    future_distance = float(np.linalg.norm(person_future - vehicle_future))
                    if future_distance < best_future_distance:
                        best_future_distance = future_distance
                        best_future_time = float(future_time)
                        best_person_future = person_future
                        best_vehicle_future = vehicle_future

                dynamic_threshold = get_dynamic_projection_threshold(vehicle, person)
                if best_future_time is None:
                    continue

                # --------------------------------------------------------
                # B-1. 접근단계 점수:
                # 기존에는 dynamic_threshold 안으로 들어와야만 점수가 발생했기 때문에
                # 화면에서 거의 충돌 직전에만 위험도가 급상승했습니다.
                # 여기서는 이미 계산된 미래 최소거리와 접근시간을 이용해,
                # 위험거리 진입 전에도 단계적으로 표시 점수를 부여합니다.
                # --------------------------------------------------------
                if ENABLE_PROGRESSIVE_APPROACH_RISK and best_future_distance >= dynamic_threshold:
                    monitor_limit = dynamic_threshold * APPROACH_MONITOR_MULTIPLIER
                    caution_limit = dynamic_threshold * APPROACH_CAUTION_MULTIPLIER

                    if best_future_distance < monitor_limit:
                        urgency = max(0.0, 1.0 - best_future_time / PROJECTION_HORIZON_S)

                        if best_future_distance < caution_limit:
                            # 위험거리 바로 바깥에 근접한 예측: CAUTION까지 자연스럽게 상승
                            band_ratio = float(np.clip(
                                (caution_limit - best_future_distance)
                                / max(caution_limit - dynamic_threshold, 1e-6),
                                0.0, 1.0
                            ))
                            approach_risk_percent = int(np.clip(
                                CAUTION_RISK_THRESHOLD
                                + (APPROACH_CAUTION_MAX_SCORE - CAUTION_RISK_THRESHOLD) * band_ratio
                                + 4.0 * urgency,
                                CAUTION_RISK_THRESHOLD,
                                APPROACH_CAUTION_MAX_SCORE
                            ))
                            approach_source = "APPROACH CAUTION"
                        else:
                            # 아직 위험거리와는 여유가 있지만 충돌 경로 쪽으로 접근 중인 상태
                            band_ratio = float(np.clip(
                                (monitor_limit - best_future_distance)
                                / max(monitor_limit - caution_limit, 1e-6),
                                0.0, 1.0
                            ))
                            approach_risk_percent = int(np.clip(
                                APPROACH_BASE_SCORE
                                + (APPROACH_MONITOR_MAX_SCORE - APPROACH_BASE_SCORE) * band_ratio
                                + 4.0 * urgency,
                                APPROACH_BASE_SCORE,
                                APPROACH_MONITOR_MAX_SCORE
                            ))
                            approach_source = "APPROACH MONITORING"

                        if approach_risk_percent > max_rule_risk_percent:
                            max_rule_risk_percent = approach_risk_percent
                            rule_source = approach_source
                            vehicle_future_px = np.array([
                                best_vehicle_future[0] * minimap_w,
                                best_vehicle_future[1] * minimap_h
                            ], dtype=np.float32)
                            person_future_px = np.array([
                                best_person_future[0] * minimap_w,
                                best_person_future[1] * minimap_h
                            ], dtype=np.float32)
                            best_projection_paths = (
                                (int(vehicle["pt_px"][0]), int(vehicle["pt_px"][1])),
                                (int(vehicle_future_px[0]), int(vehicle_future_px[1])),
                                (int(person["pt_px"][0]), int(person["pt_px"][1])),
                                (int(person_future_px[0]), int(person_future_px[1])),
                            )
                            best_rule_line = (
                                (int(vehicle_future_px[0]), int(vehicle_future_px[1])),
                                (int(person_future_px[0]), int(person_future_px[1])),
                            )
                            best_rule_text = (
                                (
                                    max(0, min(minimap_w - 340, int(vehicle_future_px[0]))),
                                    max(48, min(minimap_h - 15, int(vehicle_future_px[1]) - 15))
                                ),
                                f"TTC:{ttc:.2f}s APPROACH_D:{best_future_distance:.3f} T:{best_future_time:.2f}s",
                            )
                            best_risk_basis = {
                                "source": approach_source,
                                "distance": initial_distance,
                                "closing_speed": closing_speed,
                                "ttc": ttc,
                                "t_cpa": best_future_time,
                                "cpa_distance": best_future_distance,
                                "limit": dynamic_threshold,
                            }

                    # 아직 기존 충돌위험 threshold 안으로 들어온 것은 아니므로,
                    # 기존 고위험 산식은 실행하지 않습니다.
                    continue

                if best_future_distance >= dynamic_threshold:
                    continue

                urgency = max(0.0, 1.0 - best_future_time / PROJECTION_HORIZON_S)
                closeness = max(0.0, 1.0 - best_future_distance / dynamic_threshold)

                # 큰 차량의 차체 측면 근접을 동적 threshold로 반영합니다.
                # 경로가 겹치고 임박하며 차체 기준으로 가까울수록 점수가 증가합니다.
                projection_risk_percent = calculate_time_based_risk_percent(best_future_time)

                if projection_risk_percent > max_rule_risk_percent:
                    max_rule_risk_percent = projection_risk_percent
                    rule_source = "FORWARD PROJECTION"
                    vehicle_future_px = np.array([
                        best_vehicle_future[0] * minimap_w,
                        best_vehicle_future[1] * minimap_h
                    ], dtype=np.float32)
                    person_future_px = np.array([
                        best_person_future[0] * minimap_w,
                        best_person_future[1] * minimap_h
                    ], dtype=np.float32)
                    best_projection_paths = (
                        (int(vehicle["pt_px"][0]), int(vehicle["pt_px"][1])),
                        (int(vehicle_future_px[0]), int(vehicle_future_px[1])),
                        (int(person["pt_px"][0]), int(person["pt_px"][1])),
                        (int(person_future_px[0]), int(person_future_px[1])),
                    )
                    best_rule_line = (
                        (int(vehicle_future_px[0]), int(vehicle_future_px[1])),
                        (int(person_future_px[0]), int(person_future_px[1])),
                    )
                    best_rule_text = (
                        (
                            max(0, min(minimap_w - 340, int(vehicle_future_px[0]))),
                            max(48, min(minimap_h - 15, int(vehicle_future_px[1]) - 15))
                        ),
                        f"TTC:{ttc:.2f}s PRED_D:{best_future_distance:.3f} T:{best_future_time:.2f}s",
                    )
                    best_risk_basis = {
                        "source": "FORWARD PROJECTION",
                        "distance": initial_distance,
                        "closing_speed": closing_speed,
                        "ttc": ttc,
                        "t_cpa": best_future_time,
                        "cpa_distance": best_future_distance,
                        "limit": dynamic_threshold,
                    }

    if max_rule_risk_percent >= DANGER_RISK_THRESHOLD:
        status = "DANGER"
        color = (0, 0, 255)
    elif max_rule_risk_percent >= WARNING_RISK_THRESHOLD:
        status = "WARNING"
        color = (0, 165, 255)
    elif max_rule_risk_percent >= CAUTION_RISK_THRESHOLD:
        status = "CAUTION"
        color = (0, 255, 255)
    else:
        status = "SAFE"
        color = (0, 255, 0)
        if rule_source == "NONE":
            rule_source = "MONITORING"

    if max_rule_risk_percent >= CAUTION_RISK_THRESHOLD:
        if best_projection_paths is not None:
            vehicle_now, vehicle_future, person_now, person_future = best_projection_paths
            # 위험 판단 근거가 되는 예상 경로는 선으로만 표시하고 방향 화살표는 노출하지 않습니다.
            cv2.line(
                minimap, vehicle_now, vehicle_future, color,
                PROJECTION_TRAJECTORY_THICKNESS, cv2.LINE_AA
            )
            cv2.line(
                minimap, person_now, person_future, (255, 255, 0),
                PROJECTION_TRAJECTORY_THICKNESS, cv2.LINE_AA
            )
            cv2.line(
                minimap, vehicle_future, person_future, color,
                PROJECTION_PAIR_LINE_THICKNESS, cv2.LINE_AA
            )
        elif best_rule_line is not None:
            cv2.line(
                minimap, best_rule_line[0], best_rule_line[1],
                color, MINIMAP_RISK_LINE_THICKNESS
            )

        if best_rule_text is not None:
            cv2.putText(
                minimap, best_rule_text[1], best_rule_text[0],
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 3, cv2.LINE_AA
            )

    return status, color, max_rule_risk_percent, rule_source, best_risk_basis


def build_scene_feature(obs_vehicles, obs_pedestrians, risk_vehicles, risk_pedestrians):
    """
    Transformer용 feature는 Observation Zone 기반 흐름 + Risk Zone 상태를 사용합니다.
    RULE 위험도 자체는 AI feature에 넣지 않아 역할이 뒤섞이지 않도록 합니다.
    """
    obs_moving_vehicles = [
        vehicle for vehicle in obs_vehicles
        if float(np.linalg.norm(vehicle["vel_norm_s"])) > MIN_VEHICLE_SPEED_NORM_S
    ]
    risk_moving_vehicles = [
        vehicle for vehicle in risk_vehicles
        if float(np.linalg.norm(vehicle["vel_norm_s"])) > MIN_VEHICLE_SPEED_NORM_S
    ]

    risk_person_ids = {p["id"] for p in risk_pedestrians}
    risk_vehicle_ids = {v["id"] for v in risk_vehicles}
    waiting_person_count = sum(1 for p in obs_pedestrians if p["id"] not in risk_person_ids)
    approaching_vehicle_count = sum(1 for v in obs_moving_vehicles if v["id"] not in risk_vehicle_ids)

    obs_min_distance, obs_max_closing, obs_min_cpa, obs_min_time_to_cpa = calculate_pair_metrics(
        obs_vehicles, obs_pedestrians
    )

    feature_vector = np.array([
        float(len(obs_pedestrians)),
        float(len(obs_vehicles)),
        float(len(obs_moving_vehicles)),
        float(len(risk_pedestrians)),
        float(len(risk_vehicles)),
        float(len(risk_moving_vehicles)),
        float(waiting_person_count),
        float(approaching_vehicle_count),
        float(obs_min_distance),
        float(obs_max_closing),
        float(obs_min_cpa),
        float(obs_min_time_to_cpa),
    ], dtype=np.float32)

    return feature_vector


def save_scene_sequences(csv_path, output_dir, index_path, sequence_label, seq_len=30, stride=10):
    """매 프레임 Observation/Risk scene feature를 연속 시퀀스로 저장합니다."""
    if not csv_path.exists():
        print("[경고] Scene CSV 파일이 없습니다.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[경고] 저장된 scene feature가 없습니다.")
        return

    df = df.sort_values("frame").copy()
    df["segment"] = (df["frame"].diff().fillna(1) > MAX_FRAME_GAP).cumsum()

    index_rows = []
    saved_count = 0
    for segment_id, segment in df.groupby("segment"):
        sequence = segment[FEATURE_NAMES].to_numpy(dtype=np.float32)
        if len(sequence) < seq_len:
            continue

        for start in range(0, len(sequence) - seq_len + 1, stride):
            sample = sequence[start:start + seq_len]
            start_frame = int(segment.iloc[start]["frame"])
            end_frame = int(segment.iloc[start + seq_len - 1]["frame"])
            filename = f"{VIDEO_NAME}_obsrisk_seg{int(segment_id)}_f{start_frame}-{end_frame}.npy"
            np.save(output_dir / filename, sample)
            index_rows.append({
                "file": filename,
                "video_name": VIDEO_NAME,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "sequence_length": seq_len,
                "feature_dim": FEATURE_DIM,
                "label": sequence_label,
            })
            saved_count += 1

    if index_rows:
        pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding="utf-8-sig")

    scene_type = "정상 학습용" if sequence_label == "normal" else "시연/테스트용"
    print(f"[완료] Transformer {scene_type} Observation/Risk scene 시퀀스 {saved_count}개 저장")
    print(f"[저장 위치] {output_dir}")
    if saved_count == 0:
        print(f"[안내] 영상 길이가 {seq_len}프레임 미만인지 확인하세요.")


# ============================================================
# 4. Observation/Risk Scene-based Transformer 추론 모델
# ============================================================
@dataclass
class ModelConfig:
    feature_dim: int = FEATURE_DIM
    input_len: int = INPUT_LEN
    pred_len: int = PRED_LEN
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1


class NormalObsRiskSceneTransformer(nn.Module):
    """정상 Observation/Risk 장면 흐름의 이후 feature를 예측하는 Transformer입니다."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.feature_dim, config.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.input_len, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.pred_len * config.feature_dim),
        )

    def forward(self, x):
        h = self.input_projection(x) + self.pos_embedding
        h = self.encoder(h)
        context = h.mean(dim=1)
        prediction = self.forecast_head(context)
        return prediction.view(x.size(0), self.config.pred_len, self.config.feature_dim)


def load_transformer_for_demo():
    if RUN_MODE != "demo" or not SHOW_AI_PANEL:
        return None, None, None, None

    if not TRANSFORMER_MODEL_PATH.exists() or not TRANSFORMER_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Observation/Risk Scene Transformer 모델이 없습니다.\n"
            f"모델: {TRANSFORMER_MODEL_PATH}\n"
            "먼저 train_obsrisk_scene_transformer_30frames.py를 실행하세요."
        )

    checkpoint = torch.load(TRANSFORMER_MODEL_PATH, map_location=TRANSFORMER_DEVICE, weights_only=False)
    with open(TRANSFORMER_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    config = ModelConfig(**checkpoint["model_config"])
    transformer = NormalObsRiskSceneTransformer(config).to(TRANSFORMER_DEVICE)
    transformer.load_state_dict(checkpoint["model_state_dict"])
    transformer.eval()

    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    threshold = float(metadata["anomaly_threshold_mse"])
    print(f"[Observation/Risk Scene Transformer 로드 완료] threshold(MSE) = {threshold:.6f}")
    return transformer, mean, std, threshold


@torch.no_grad()
def calculate_anomaly_score(sequence, transformer, mean, std):
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.shape != (SEQ_LEN, FEATURE_DIM):
        return None

    normalized = (arr - mean[None, :]) / std[None, :]
    x = torch.from_numpy(normalized[:INPUT_LEN]).unsqueeze(0).to(TRANSFORMER_DEVICE)
    y_true = torch.from_numpy(normalized[INPUT_LEN:INPUT_LEN + PRED_LEN]).unsqueeze(0).to(TRANSFORMER_DEVICE)
    y_pred = transformer(x)
    return float(torch.mean((y_pred - y_true) ** 2).item())


# ============================================================
# 5. 실행 준비
# ============================================================
if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {VIDEO_PATH}")
if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO 모델 파일을 찾을 수 없습니다: {YOLO_MODEL_PATH}")

yolo_model = YOLO(str(YOLO_MODEL_PATH))
src_pts, dst_pts, minimap_w, minimap_h = get_homography_setup()

# 6개 대응점을 반영해 하나의 도로 평면 Homography를 계산합니다.
homography_matrix, homography_status = cv2.findHomography(src_pts, dst_pts, method=0)
if homography_matrix is None:
    raise RuntimeError("6점 Homography 계산에 실패했습니다. 설정을 다시 진행하세요.")
inverse_homography_matrix = np.linalg.inv(homography_matrix)

# 사용자가 클릭한 6점 경계를 변환한 Polygon 자체가 실제 Risk Zone입니다.
RISK_POLYGON_BEV = cv2.perspectiveTransform(
    np.array([src_pts], dtype=np.float32), homography_matrix
)[0]
RISK_POLYGON_NORM = np.array([
    [point[0] / max(minimap_w, 1), point[1] / max(minimap_h, 1)]
    for point in RISK_POLYGON_BEV
], dtype=np.float32)

# OBS는 6점 Risk Polygon을 중심점 기준으로 바깥쪽으로 확장한 관찰영역입니다.
risk_centroid_norm = np.mean(RISK_POLYGON_NORM, axis=0)
obs_expansion_scale = 1.0 + 2.0 * OBS_MARGIN
OBS_POLYGON_NORM = risk_centroid_norm + (
    RISK_POLYGON_NORM - risk_centroid_norm
) * obs_expansion_scale
OBS_POLYGON_BEV = np.array([
    [point[0] * minimap_w, point[1] * minimap_h]
    for point in OBS_POLYGON_NORM
], dtype=np.float32)

transformer_model, feature_mean, feature_std, anomaly_threshold = load_transformer_for_demo()
scene_buffer = deque(maxlen=SEQ_LEN)
kalman_filters = {}
# RULE 판정 전용 차량 이동상태 누적 저장소: Transformer 입력에는 사용하지 않음
rule_vehicle_motion_states = {}

print(f"\n[실행 모드] {RUN_MODE}")
print(f"[입력 영상] {VIDEO_PATH}")
print(f"[Observation Zone] 6점 Risk Polygon 주변 {OBS_MARGIN * 100:.0f}% 확장: 접근 차량 관찰용")
print("[Risk Zone] 사용자가 지정한 6점 Polygon 내부: TTC-like + CPA RULE 판정용")
print("[Forward Projection] Risk 보행자 + Observation 접근 차량: 미래 경로 사전 위험점수 활성화")
print("[Progressive Approach Risk] 위험거리 진입 전 접근단계 점수 표시 활성화")
print("[Dynamic Footprint] Forward Projection에 BEV상 차량/보행자 폭 반영")
print("[RULE Motion Filter] 연속 이동이 확인된 차량만 위험계산 참여")
print("[STOP-OCC Guard] 정차 차량의 겹침/가림 구간은 MOVING 승격 차단 (화면 비표시)")
print("[Display] 내부 추적 ID와 이동상태는 유지하되 화면에는 person/car 및 OBS만 표시")
print(f"[AI Panel] {'ON' if SHOW_AI_PANEL else 'OFF'} - 기본 시연 화면은 RULE 위험판정 중심")
print("[Kalman Filter] 호모그래피 변환 후 객체별 위치·속도 평활화 유지")
print(f"[Display Smoothing] 위험도 표시 ON: rise +{DISPLAY_RISK_RISE_STEP_PER_FRAME:.0f}/frame, fall -{DISPLAY_RISK_FALL_STEP_PER_FRAME:.0f}/frame")
print(f"[시퀀스] {SEQ_LEN} frames = input {INPUT_LEN} + prediction {PRED_LEN}")


# ============================================================
# 6. 영상 처리
# ============================================================
cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

# CSV에는 Transformer 입력 feature뿐 아니라 화면에 표시되는 RULE/TTC 계산 근거도 함께 저장합니다.
# FEATURE_NAMES는 기존 Transformer 시퀀스 생성에 그대로 사용되며, 아래 추가 컬럼은 검증/발표용 로그입니다.
CSV_RISK_LOG_HEADERS = [
    "rule_status",
    "rule_risk_percent",
    "rule_source",
    "raw_rule_status",
    "raw_rule_risk_percent",
    "raw_rule_source",
    "current_distance_norm",
    "closing_speed_norm_s",
    "ttc_s",
    "t_cpa_s",
    "cpa_distance_norm",
    "risk_limit_norm",
]

csv_headers = ["video_name", "frame"] + FEATURE_NAMES + CSV_RISK_LOG_HEADERS
frame_num = 0

# 좌측 상단 패널에만 사용하는 평활화된 표시 점수
# 실제 위험판정 계산값과 분리되어 있어 RULE 로직에는 영향을 주지 않습니다.
display_risk_score = 0.0
display_rule_source = "MONITORING"
display_risk_basis = None

# 기존 표시 방식 유지: 화면 오른쪽 잘림 방지
DISPLAY_WINDOW_NAME = "Tracking"
cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(DISPLAY_WINDOW_NAME, 1600, 900)

with open(SCENE_CSV_FILE, "w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_num += 1

        # 검은 배경 점 표시형 6점 Risk BEV 미니맵
        minimap = np.zeros((minimap_h, minimap_w, 3), dtype=np.uint8)
        cv2.polylines(
            minimap, [np.int32(np.round(RISK_POLYGON_BEV))],
            True, (80, 80, 80), 4, cv2.LINE_AA
        )
        obs_vehicles = []
        obs_pedestrians = []
        risk_vehicles = []
        risk_pedestrians = []

        results = yolo_model.track(
            frame,
            persist=True,
            conf=DETECTION_CONF,
            iou=TRACKING_IOU,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(xyxy, track_ids, classes):
                class_name = yolo_model.names[int(cls)]
                if class_name not in VEHICLE_CLASSES and class_name not in PEDESTRIAN_CLASSES:
                    continue

                # 내부 track_id는 추적/Kalman 상태 연결에 계속 사용하지만 화면에는 표시하지 않습니다.
                display_object_name = "person" if class_name in PEDESTRIAN_CLASSES else "car"
                bottom_center = np.array(
                    [[[(box[0] + box[2]) / 2.0, float(box[3])]]], dtype=np.float32
                )
                mapped = cv2.perspectiveTransform(bottom_center, homography_matrix)[0][0]

                # Kalman Filter 유지: Homography 변환된 좌표와 속도 안정화
                if track_id not in kalman_filters:
                    kalman_filters[track_id] = make_kalman_filter(mapped)

                kf = kalman_filters[track_id]
                kf.predict()
                estimated = kf.correct(np.array([[mapped[0]], [mapped[1]]], dtype=np.float32))
                smoothed_pt_px = np.array([estimated[0][0], estimated[1][0]], dtype=np.float32)
                velocity_px_per_frame = np.array([estimated[2][0], estimated[3][0]], dtype=np.float32)

                pt_norm = np.array([
                    smoothed_pt_px[0] / max(minimap_w, 1),
                    smoothed_pt_px[1] / max(minimap_h, 1),
                ], dtype=np.float32)
                vel_norm_s = np.array([
                    velocity_px_per_frame[0] * fps / max(minimap_w, 1),
                    velocity_px_per_frame[1] * fps / max(minimap_h, 1),
                ], dtype=np.float32)

                inside_risk = is_inside_risk(pt_norm)
                inside_observation = is_inside_observation(pt_norm)

                width_norm = calculate_bev_footprint_width_norm(
                    box, homography_matrix, minimap_w, minimap_h
                )

                record = {
                    "id": int(track_id),
                    "pt_px": smoothed_pt_px,
                    "pt_norm": pt_norm,
                    "vel_norm_s": vel_norm_s,
                    "width_norm": width_norm,
                    "class_name": class_name,
                }

                if class_name in PEDESTRIAN_CLASSES:
                    base_color = (255, 255, 0)
                    # 보행자는 기존 Kalman 보정 속도를 화면에 그대로 표시합니다.
                    record["display_vel_norm_s"] = record["vel_norm_s"].copy()
                else:
                    base_color = (0, 255, 0)
                    # Observation/Risk Zone 안의 차량만 RULE 이동상태를 누적합니다.
                    # OUT 차량은 현재 RULE 계산 대상이 아니므로 상태 갱신이 필요 없습니다.
                    if inside_observation:
                        existing_state = rule_vehicle_motion_states.get(int(track_id))
                        already_moving = bool(existing_state and existing_state.get("moving", False))
                        overlap_guard = False
                        if not already_moving:
                            overlap_guard = is_stopped_vehicle_occluded(
                                box, track_id, xyxy, track_ids, classes, yolo_model.names
                            )
                        update_rule_vehicle_motion_state(
                            record, rule_vehicle_motion_states,
                            block_stop_transition=overlap_guard
                        )
                    else:
                        record["rule_moving"] = False
                        record["rule_vel_norm_s"] = np.zeros(2, dtype=np.float32)
                        record["rule_speed_norm_s"] = 0.0
                        record["display_vel_norm_s"] = np.zeros(2, dtype=np.float32)

                if inside_risk:
                    display_color = base_color
                    # RULE 내부에서는 차량 이동/정지 상태를 계속 사용하지만 화면에는 노출하지 않습니다.
                    display_label = display_object_name
                    if class_name in PEDESTRIAN_CLASSES:
                        risk_pedestrians.append(record)
                        obs_pedestrians.append(record)
                    else:
                        risk_vehicles.append(record)
                        obs_vehicles.append(record)

                    x_pt, y_pt = int(smoothed_pt_px[0]), int(smoothed_pt_px[1])
                    # 축소된 화면에서도 쉽게 구분되도록 점을 크게 표시
                    cv2.circle(
                        minimap, (x_pt, y_pt),
                        MINIMAP_OUTLINE_RADIUS, (255, 255, 255), MINIMAP_OUTLINE_THICKNESS
                    )
                    cv2.circle(
                        minimap, (x_pt, y_pt),
                        MINIMAP_POINT_RADIUS, base_color, -1
                    )
                    short_label = "PERSON" if class_name in PEDESTRIAN_CLASSES else "CAR"
                    cv2.putText(
                        minimap, short_label, (x_pt + MINIMAP_OUTLINE_RADIUS + 6, y_pt + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, MINIMAP_LABEL_FONT_SCALE,
                        (255, 255, 255), MINIMAP_LABEL_THICKNESS, cv2.LINE_AA
                    )
                    # Risk Zone 내부 객체만 정규화 속도 표시
                    draw_velocity_vector_on_minimap(minimap, record, base_color)
                elif inside_observation:
                    # Observation Zone 객체는 구역 여부만 표시하고 이동/정지 상태는 숨깁니다.
                    display_color = (0, 165, 255)  # orange
                    display_label = f"{display_object_name}[OBS]"
                    if class_name in PEDESTRIAN_CLASSES:
                        obs_pedestrians.append(record)
                    else:
                        obs_vehicles.append(record)
                else:
                    display_color = (160, 160, 160)
                    display_label = f"{display_object_name}[OUT]"

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), display_color, 2)
                cv2.putText(
                    frame, display_label, (box[0], max(box[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA
                )

                # RULE 계산 대상인 Risk Zone 내부 객체에만 원본 화면 벡터를 표시
                if inside_risk:
                    draw_velocity_vector_on_frame(
                        frame, record, box, base_color,
                        inverse_homography_matrix, minimap_w, minimap_h
                    )

        cv2.putText(
            minimap, "Velocity: norm/s", (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 255), 3, cv2.LINE_AA
        )

        scene_vector = build_scene_feature(
            obs_vehicles, obs_pedestrians, risk_vehicles, risk_pedestrians
        )
        rule_status, rule_color, risk_percent, rule_source, risk_basis = calculate_rule_risk(
            risk_vehicles, risk_pedestrians, obs_vehicles, minimap
        )

        # 내부 계산 결과(risk_percent)는 그대로 유지하고 디스플레이 값만 평활화합니다.
        display_risk_score = update_display_risk_score(display_risk_score, risk_percent)
        display_rule_status, display_rule_color = get_display_risk_state(display_risk_score)

        # 점수가 서서히 내려가는 동안 SOURCE가 갑자기 MONITORING으로 바뀌지 않도록 유지합니다.
        if risk_percent > 0 and rule_source not in {"NONE", "MONITORING"}:
            display_rule_source = rule_source
            display_risk_basis = risk_basis
        elif display_risk_score <= 0.0:
            display_rule_source = "MONITORING"
            display_risk_basis = None

        # 화면 패널에 표시되는 위험도 계산 근거를 CSV에 함께 저장합니다.
        # 값이 없는 프레임은 NaN으로 저장되어 엑셀/판다스에서 빈 값처럼 다룰 수 있습니다.
        if display_risk_basis is not None:
            current_distance_norm = display_risk_basis.get("distance", np.nan)
            closing_speed_norm_s = display_risk_basis.get("closing_speed", np.nan)
            ttc_s = display_risk_basis.get("ttc", np.nan)
            t_cpa_s = display_risk_basis.get("t_cpa", np.nan)
            cpa_distance_norm = display_risk_basis.get("cpa_distance", np.nan)
            risk_limit_norm = display_risk_basis.get("limit", np.nan)
        else:
            current_distance_norm = np.nan
            closing_speed_norm_s = np.nan
            ttc_s = np.nan
            t_cpa_s = np.nan
            cpa_distance_norm = np.nan
            risk_limit_norm = np.nan

        writer.writerow([
            VIDEO_NAME,
            frame_num,
            *scene_vector.tolist(),
            display_rule_status,
            float(display_risk_score),
            display_rule_source,
            rule_status,
            float(risk_percent),
            rule_source,
            current_distance_norm,
            closing_speed_norm_s,
            ttc_s,
            t_cpa_s,
            cpa_distance_norm,
            risk_limit_norm,
        ])
        scene_buffer.append(scene_vector)

        ai_score = None
        ai_state = "COLLECTING"
        ai_color = (160, 160, 160)

        if (
            SHOW_AI_PANEL
            and RUN_MODE == "demo"
            and transformer_model is not None
            and len(scene_buffer) == SEQ_LEN
        ):
            ai_score = calculate_anomaly_score(
                list(scene_buffer), transformer_model, feature_mean, feature_std
            )
            if ai_score is not None:
                if ai_score > anomaly_threshold:
                    ai_state = "ANOMALY DETECTED"
                    ai_color = (0, 0, 255)
                else:
                    ai_state = "NORMAL"
                    ai_color = (0, 255, 0)

        frame = draw_rule_panel(
            frame, display_rule_status, display_rule_color,
            display_risk_score, display_rule_source
        )
        frame = draw_risk_basis_panel(frame, display_risk_basis)
        if SHOW_AI_PANEL and RUN_MODE == "demo":
            frame = draw_ai_panel(
                frame, ai_state, ai_color, ai_score,
                anomaly_threshold, len(scene_buffer)
            )
        frame = draw_scene_summary(frame, scene_vector)

        small_minimap = fit_minimap_to_square(minimap, canvas_size=MINIMAP_CANVAS_SIZE)
        y_off = frame.shape[0] - MINIMAP_CANVAS_SIZE - 20
        x_off = frame.shape[1] - MINIMAP_CANVAS_SIZE - 20
        if y_off >= 0 and x_off >= 0:
            roi_area = frame[
                y_off:y_off + MINIMAP_CANVAS_SIZE,
                x_off:x_off + MINIMAP_CANVAS_SIZE
            ]
            frame[
                y_off:y_off + MINIMAP_CANVAS_SIZE,
                x_off:x_off + MINIMAP_CANVAS_SIZE
            ] = cv2.addWeighted(
                roi_area, 1.0 - MINIMAP_OPACITY,
                small_minimap, MINIMAP_OPACITY, 0
            )

        cv2.imshow(DISPLAY_WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()


# ============================================================
# 7. 프로그램 종료 후 Scene Sequence 생성
# ============================================================
print("\n[시스템 종료] Transformer용 Observation/Risk scene 시퀀스를 생성합니다...")
save_scene_sequences(
    SCENE_CSV_FILE,
    VIDEO_SEQUENCE_DIR,
    SEQUENCE_INDEX_FILE,
    sequence_label=DATA_LABEL,
    seq_len=SEQ_LEN,
    stride=STRIDE,
)
print(f"[Scene CSV 저장] {SCENE_CSV_FILE}")
print(f"[시퀀스 저장] {VIDEO_SEQUENCE_DIR}")

if RUN_MODE == "normal":
    print("[정상 데이터 수집 완료] 진입 전 관찰 맥락을 포함한 정상 Observation/Risk scene 데이터입니다.")
else:
    if SHOW_AI_PANEL:
        print("[demo 완료] AI 이상점수와 RULE 위험도를 각각 확인하세요.")
    else:
        print("[demo 완료] RULE 위험도, 판단 SOURCE 및 정규화 속도를 확인하세요.")