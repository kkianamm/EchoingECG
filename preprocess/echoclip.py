import open_clip
import torch
from torchvision import transforms


def grab_echo_clip_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ]
    )


@torch.no_grad()
def process_echofeatures(model: torch.nn.Module, video: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    video = video.to(device)
    output = model.encode_image(video)  # type: ignore
    output /= output.norm(dim=-1, keepdim=True)
    return output.cpu()


def extract_vision_echoclip(savepath: str, pretrained_path: str) -> None:
    model = open_clip.create_model(
        "convnext_base",
        pretrained=pretrained_path,
    )
    vision_model = model.visual
    torch.save(vision_model.state_dict(), savepath)
    print(f"save successful in {savepath}")

    vision_model2 = open_clip.create_model("convnext_base").visual
    vision_model2.load_state_dict(torch.load(savepath))
    print(f"load successful from {savepath}")


def load_video_frames(avi_path: str) -> torch.Tensor:
    """Load an .avi video into a tensor of shape (T, 3, 224, 224).

    - Assumes each frame is already 224x224 per the user's data constraint.
    - Converts BGR (OpenCV) to RGB and scales to [0, 1] float32.
    """
    try:
        import cv2  # Lazy import to keep module import light
    except ImportError as e:
        raise RuntimeError("opencv-python is required to read AVI files. Please install it.") from e

    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {avi_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame is BGR HxWxC
        h, w, c = frame.shape
        assert h == 224 and w == 224, f"Expected 224x224 frames, got {h}x{w} for {avi_path}"
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To tensor CxHxW in [0,1]
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float() / 255.0
        frames.append(frame_t)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {avi_path}")

    return torch.stack(frames, dim=0)  # (T, 3, 224, 224)
