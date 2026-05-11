"""
Inference server that drives the Udacity simulator with a trained PilotNet.

When the Udacity Self-Driving Car Simulator runs in Autonomous Mode, it
opens a WebSocket connection to localhost:4567 and streams camera frames.
For every frame, we run our trained model and send back a steering command
and a throttle target.

Usage:
    python -m src.drive --model checkpoints/best.pth

Then in a separate window: open the simulator, select a track,
and click "Autonomous Mode".
"""

import argparse
import base64
from io import BytesIO

import cv2
import eventlet
import numpy as np
import socketio
import torch
from flask import Flask
from PIL import Image

from src.dataset import preprocess
from src.model import PilotNet


# Target speed in mph. The PilotNet model only predicts steering, not throttle;
# we use a simple proportional controller to maintain this speed.
TARGET_SPEED = 15.0


def load_model(checkpoint_path: str, device: torch.device) -> PilotNet:
    """Load a trained PilotNet from a state_dict checkpoint and put it on the device in eval mode."""
    model = PilotNet().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Module-level objects so the socketio handlers can reach them.
sio = socketio.Server()
app = Flask(__name__)
MODEL: PilotNet = None
DEVICE: torch.device = None


@sio.on("connect")
def on_connect(sid, environ):
    print(f"Simulator connected (sid={sid})")
    send_control(steering_angle=0.0, throttle=0.0)


@sio.on("disconnect")
def on_disconnect(sid):
    print(f"Simulator disconnected (sid={sid})")


@sio.on("telemetry")
def on_telemetry(sid, data):
    """Called every frame the simulator sends us in Autonomous Mode."""
    if data is None:
        sio.emit("manual", data={}, skip_sid=True)
        return

    # The simulator sends a base64-encoded JPEG of the center-camera image.
    image_b64 = data["image"]
    pil_image = Image.open(BytesIO(base64.b64decode(image_b64)))
    rgb_array = np.asarray(pil_image)
    # PIL gives RGB; our preprocess() expects BGR (it was built around cv2).
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # Use the SAME preprocessing as training -- crop, resize, RGB, transpose.
    processed = preprocess(bgr_array)
    tensor = torch.from_numpy(processed).unsqueeze(0).to(DEVICE)  # add batch dim

    with torch.no_grad():
        steering_angle = MODEL(tensor).item()

    # Simple proportional throttle: speed up if below target, ease off if above.
    current_speed = float(data["speed"])
    throttle = 0.3 if current_speed < TARGET_SPEED else 0.0

    send_control(steering_angle, throttle)


def send_control(steering_angle: float, throttle: float):
    """Push a steering+throttle command to the simulator."""
    sio.emit("steer", data={
        "steering_angle": str(steering_angle),
        "throttle": str(throttle),
    }, skip_sid=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/best.pth",
                        help="Path to a trained PilotNet checkpoint (.pth).")
    parser.add_argument("--port", type=int, default=4567,
                        help="Port for the simulator to connect to. Default matches Udacity sim.")
    args = parser.parse_args()

    global MODEL, DEVICE
    DEVICE = pick_device()
    MODEL = load_model(args.model, DEVICE)
    print(f"Loaded model from {args.model}")
    print(f"Inference device: {DEVICE}")
    print(f"Listening on port {args.port}. Start the simulator in Autonomous Mode.")

    # Wrap the Flask app with socketio middleware and serve with eventlet.
    wsgi_app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", args.port)), wsgi_app)


if __name__ == "__main__":
    main()