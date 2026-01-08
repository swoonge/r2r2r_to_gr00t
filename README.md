# r2r2r_to_gr00t

Real2Render2Real(r2r2r) 데이터를 GR00T에서 사용하는 LeRobot v2 포맷으로 변환하는 도구입니다.

---

## 1. 개요

이 레포지토리는 r2r2r에서 생성된 YuMi 데모 데이터를 GR00T 학습 파이프라인에 바로 연결할 수 있도록,
LeRobot v2 + GR00T 확장 메타 구조(`meta/`, `data/`, `videos/`)로 변환합니다.

핵심 목표는 다음과 같습니다.
- r2r2r 원본 데이터의 구조를 유지하면서 GR00T 입력 규격에 맞게 재구성
- GR00T 학습을 위해 필요한 `stats.json`, `relative_stats.json` 생성 가능
- 변환 옵션은 CLI로 제어 가능하되 기본값은 합리적인 기준(해상도는 GR00T demo, fps는 r2r2r)으로 설정
- YuMi / Franka 데이터 모두 동일한 변환 파이프라인으로 처리

---

## 2. 입력 데이터 구조 (r2r2r: yumi_coffee_maker)

기본적으로 다음 경로 구조를 가정합니다.

```
yumi_coffee_maker/
  successes/
    env_0_YYYY_MM_DD_HH_MM_SS/
      camera_0/rgb/0000.jpg ...
      camera_1/rgb/0000.jpg ...
      robot_data/robot_data.h5
      robot_data/joint_names.txt
```

### 입력 데이터 요약
- 이미지 포맷: JPG
- 해상도: 1280x720
- fps: 15 (r2r2r 변환 스크립트 기준)
- 프레임 길이: 약 161 프레임/에피소드
- HDF5 (`robot_data.h5`):
  - `joint_angles`: (T, 16) float32
  - `ee_poses`: (T, 14) float32
  - `gripper_binary_cmd`: (T, 2) bool
- `joint_names.txt`에 16개 관절 이름 순서가 정의됨

---

## 2-1. 입력 데이터 구조 (r2r2r: franka_coffee_maker)

```
franka_coffee_maker/
  successes/
    env_0_YYYY_MM_DD_HH_MM_SS/
      camera_0/rgb/0000.jpg ...
      camera_1/rgb/0000.jpg ...
      robot_data/robot_data.h5
      robot_data/joint_names.txt
```

### 입력 데이터 요약
- 이미지 포맷: JPG
- 해상도: 1280x720 (샘플 기준)
- fps: 15 (r2r2r 변환 스크립트 기준)
- 프레임 길이: 약 90 프레임/에피소드
- HDF5 (`robot_data.h5`):
  - `joint_angles`: (T, 8) float32
  - `ee_poses`: (T, 7) float32
  - `gripper_binary_cmd`: (T, 1) bool
- `joint_names.txt`에 8개 관절 이름 순서가 정의됨
  - 예: `panda_joint1` ~ `panda_joint7`, `panda_finger_joint1`

---

## 3. 출력 데이터 구조 (GR00T / LeRobot v2)

변환 결과는 다음 구조로 생성됩니다.

```
converted_yumi/
  meta/
    info.json
    episodes.jsonl
    tasks.jsonl
    modality.json
    stats.json
    relative_stats.json
  data/
    chunk-000/episode_000000.parquet
  videos/
    chunk-000/observation.images.front/episode_000000.mp4
    chunk-000/observation.images.wrist/episode_000000.mp4
```

### meta 파일 설명
- `info.json`: 데이터 경로 패턴, fps, feature 스키마
- `episodes.jsonl`: 각 에피소드 길이 및 task 리스트
- `tasks.jsonl`: task_index -> 텍스트 매핑
- `modality.json`: state/action/video/annotation 슬라이스 및 매핑
- `stats.json`, `relative_stats.json`: GR00T에서 normalization에 사용되는 통계값

---

## 4. 변환 규칙 요약

### 4.1 에피소드 매핑
- `successes/env_0_*` 폴더 1개 = GR00T 에피소드 1개
- 길이: `T-1` (action 정의를 위해 마지막 프레임 제외)

### 4.2 State / Action
- 입력: `robot_data.h5/joint_angles`
- 출력 컬럼:
  - `observation.state`: float32, shape [N]
  - `action`: float32, shape [N]
- action은 **다음 스텝 절대값**으로 저장하고,
  GR00T 설정에서 arm은 **relative**, gripper는 **absolute**로 해석하도록 구성합니다.

**YuMi**
- 입력: (T, 16)
- 순서 재정렬: left arm (7) + right arm (7) + gripper (2)
- 출력 shape: [16]

**Franka**
- 입력: (T, 8)
- 순서 재정렬: arm (7) + gripper (1)  *(finger/gripper 키워드 기준)*
- 출력 shape: [8]

### 4.3 Video
- 입력: `camera_0` → `front`, `camera_1` → `wrist`
- 기본 fps: **15** (r2r2r 기준)
- 기본 해상도: **640x480** (GR00T demo와 동일한 출력 크기)
  - 비율 유지 + padding
  - 원본 유지 시 `--resize 0` 또는 `--resize keep`

### 4.4 Language
- r2r2r 원본에는 언어 라벨이 없음
- 단일 고정 instruction 문장을 모든 frame에 사용
- `tasks.jsonl`에 1개 task만 존재하며 `task_index=0`으로 통일

---

## 5. 사용 방법

### 5.1 설치/환경
Python 3.10 이상을 권장합니다.

```bash
pip install -r requirements.txt
```

추가로 ffmpeg가 필요합니다.
```bash
# Ubuntu 예시
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### 5.2 변환 실행
```bash
python convert_yumi_to_gr00t.py \
  --robot yumi \
  --input-root /home/vision/Sim2Real_Data_Augmentation_for_VLA/yumi_coffee_maker/successes \
  --output-root /home/vision/Sim2Real_Data_Augmentation_for_VLA/r2r2r_to_gr00t/converted_yumi \
  --task "put the white cup on the coffee machine" \
  --fps 15 \
  --resize 640x480
```

```bash
python convert_yumi_to_gr00t.py \
  --robot franka \
  --input-root /home/vision/Sim2Real_Data_Augmentation_for_VLA/franka_coffee_maker/successes \
  --output-root /home/vision/Sim2Real_Data_Augmentation_for_VLA/r2r2r_to_gr00t/converted_franka_fps15 \
  --task "put the white cup on the coffee machine" \
  --fps 15 \
  --resize 640x480
```

### 5.3 주요 옵션
- `--robot`: `yumi` 또는 `franka` (기본 `yumi`)
- `--fps`: 비디오 fps 및 timestamp 기준 (기본 15)
- `--resize`:
  - `640x480` (기본)
  - `224` (정사각형, aspect 유지 + padding)
  - `0` 또는 `keep` (원본 크기 유지)
- `--camera-keys`: 입력 카메라 폴더명 (기본 `camera_0 camera_1`)
- `--camera-names`: 출력 카메라명 (기본 `front wrist`)
- `--overwrite`: 출력 디렉토리 존재 시 덮어쓰기

---

## 6. GR00T 학습을 위한 후처리

GR00T 데이터로더는 `meta/stats.json`과 `meta/relative_stats.json`을 요구합니다.
변환 직후에는 stats가 없으므로 아래 스크립트로 생성하세요.
stats 생성 시에도 해당 모달리티 설정(예: `NEW_EMBODIMENT`, `R2R2R_FRANKA`)이 등록되어 있어야 합니다.

```bash
# GR00T 레포에서 실행
python gr00t/data/stats.py \
  --dataset-path /path/to/converted_yumi \
  --embodiment-tag NEW_EMBODIMENT

python gr00t/data/stats.py \
  --dataset-path /path/to/converted_franka_fps15 \
  --embodiment-tag R2R2R_FRANKA
```

### GR00T 모달리티 설정 예시
아래 예시는 YuMi/Franka용 설정입니다.
(예시 파일을 GR00T에서 import되도록 등록하거나, `--modality-config-path`로 지정하여 finetune을 실행하세요.)

```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["left_arm", "right_arm", "gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

```python
# Franka (r2r2r_franka)
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["arm", "gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(config, embodiment_tag=EmbodimentTag.R2R2R_FRANKA)
```

---

## 7. 출력 구조 예시
```
converted_yumi/
  meta/
    info.json
    episodes.jsonl
    tasks.jsonl
    modality.json
    stats.json
    relative_stats.json
  data/
    chunk-000/episode_000000.parquet
  videos/
    chunk-000/observation.images.front/episode_000000.mp4
    chunk-000/observation.images.wrist/episode_000000.mp4
```

---

## 8. 주의사항 / 트러블슈팅

- **frame count mismatch**: 카메라 프레임 수와 `joint_angles` 길이가 다르면 변환이 실패합니다.
- **출력 디렉토리 비어있지 않음**: `--overwrite`를 사용하세요.
- **ffmpeg 없음**: mp4 인코딩이 실패합니다. ffmpeg 설치 필요.
- **stats.json 없음**: GR00T 로더에서 에러가 발생합니다. `gr00t/data/stats.py`로 생성하세요.

---

## 9. 참고
- GR00T 데이터 준비: `Isaac-GR00T/getting_started/data_preparation.md`
- GR00T 모달리티 설정: `Isaac-GR00T/getting_started/data_config.md`
