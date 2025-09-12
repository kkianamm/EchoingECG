import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers import convert_color_space


def get_video_from_dicom(file_path: str) -> np.ndarray | None:
    try:
        dicom_data = pydicom.dcmread(file_path)
        minx = dicom_data[(0x0018, 0x6011)][0][(0x0018, 0x6018)].value
        miny = dicom_data[(0x0018, 0x6011)][0][(0x0018, 0x601A)].value
        maxx = dicom_data[(0x0018, 0x6011)][0][(0x0018, 0x601C)].value
        maxy = dicom_data[(0x0018, 0x6011)][0][(0x0018, 0x601E)].value
        images_rgb = convert_color_space(
            dicom_data.pixel_array, "YBR_FULL_422", "RGB", per_frame=True
        )
        if len(images_rgb.shape) > 3:
            # image = normalize_video_per_channel(images_rgb)
            resized_video = resize_video(images_rgb[:, miny:maxy, minx:maxx], 224, 224)
            return resized_video.astype(np.uint8)
        else:
            return None
    except Exception as e:
        print(f"Error processing DICOM file {file_path}: {e}")
        return None


def resize_video(video: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    num_frames, height, width, channels = video.shape
    resized_video = np.zeros((num_frames, target_height, target_width, channels), dtype=np.uint8)

    for frame_idx in range(num_frames):
        frame = video[frame_idx]
        resized_frame = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        resized_video[frame_idx] = resized_frame

    return resized_video


def process_dicom(
    row: dict,
) -> tuple[str, np.ndarray] | tuple[str, None]:
    file_path = row["dicom_filepath"]
    # If 'row' is a numpy array, use .item() or proper indexing; if it's a dict or pandas Series, keep as is.
    study_id = row["newidentifier"]

    # Process the DICOM file
    video = get_video_from_dicom(file_path)

    if video is not None:
        return (study_id, video)
    else:
        return (study_id, None)
