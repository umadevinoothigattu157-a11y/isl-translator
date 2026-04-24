from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os
import math

app = FastAPI(title="ISL Translator API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../dataset/isl_images")
if os.path.exists(DATASET_PATH):
    app.mount("/images", StaticFiles(directory=DATASET_PATH), name="images")

class Landmark(BaseModel):
    x: float
    y: float
    z: float

class PredictRequest(BaseModel):
    landmarks: List[Landmark]
    landmarks2: Optional[List[Landmark]] = None  # second hand

class PredictResponse(BaseModel):
    letter: str
    confidence: float
    method: str

class TextToSignRequest(BaseModel):
    text: str

class TextToSignResponse(BaseModel):
    letters: List[str]
    image_urls: List[str]

# ── Geometry helpers ──

def dist(lm, a, b):
    return math.sqrt((lm[a].x-lm[b].x)**2+(lm[a].y-lm[b].y)**2+(lm[a].z-lm[b].z)**2)

def dist_pts(p1, p2):
    return math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)

def is_extended(lm, tip, pip, mcp):
    return lm[tip].y < lm[pip].y and lm[pip].y < lm[mcp].y

def is_curled(lm, tip, dip):
    return lm[tip].y > lm[dip].y

def all_curled(lm):
    return (lm[8].y > lm[6].y + 0.02 and
            lm[12].y > lm[10].y + 0.02 and
            lm[16].y > lm[14].y + 0.02 and
            lm[20].y > lm[18].y + 0.02)

def is_o_shape(lm):
    d_ti = dist(lm, 4, 8)
    d_tm = dist(lm, 4, 12)
    d_tr = dist(lm, 4, 16)
    tips_near = d_ti < 0.10 and d_tm < 0.13 and d_tr < 0.15
    index_bent  = lm[8].y  > lm[5].y  - 0.05
    middle_bent = lm[12].y > lm[9].y  - 0.05
    ring_bent   = lm[16].y > lm[13].y - 0.05
    return tips_near and index_bent and middle_bent and ring_bent

def is_e_shape(lm):
    index_h  = lm[8].y  > lm[7].y  + 0.01
    middle_h = lm[12].y > lm[11].y + 0.01
    ring_h   = lm[16].y > lm[15].y + 0.01
    pinky_h  = lm[20].y > lm[19].y + 0.01
    not_o    = dist(lm, 4, 8) > 0.08
    thumb_low = lm[4].y > lm[5].y - 0.02
    return index_h and middle_h and ring_h and pinky_h and not_o and thumb_low

def classify_one_hand(lm):
    """Classify gesture from single hand landmarks."""

    thumb  = lm[4].x < lm[3].x
    index  = is_extended(lm, 8,  6,  5)
    middle = is_extended(lm, 12, 10, 9)
    ring   = is_extended(lm, 16, 14, 13)
    pinky  = is_extended(lm, 20, 18, 17)

    d_ti      = dist(lm, 4, 8)
    d_tm      = dist(lm, 4, 12)
    d_tr      = dist(lm, 4, 16)
    spread_im = abs(lm[8].x - lm[12].x)
    index_dx  = lm[8].x - lm[5].x
    index_dy  = lm[8].y - lm[5].y

    # O - check first (specific shape)
    if is_o_shape(lm):
        return 'O', 0.90

    # E - check before fist letters
    if is_e_shape(lm):
        return 'E', 0.88

    # A - fist thumb to side
    if not index and not middle and not ring and not pinky:
        if all_curled(lm) and lm[4].y < lm[3].y and lm[4].x < lm[5].x:
            return 'A', 0.86

    # B - four fingers up thumb tucked
    if index and middle and ring and pinky and not thumb:
        return 'B', 0.90

    # C - curved C shape
    if not index and not middle and not ring and not pinky:
        if 0.08 < d_ti < 0.16 and not all_curled(lm):
            return 'C', 0.78

    # D - index up thumb touches middle
    if index and not middle and not ring and not pinky:
        if d_tm < 0.07:
            return 'D', 0.84

    # F - thumb index touch others up
    if d_ti < 0.06 and middle and ring and pinky:
        return 'F', 0.88

    # G - index sideways
    if index and thumb and not middle and not ring and not pinky:
        if abs(index_dx) > abs(index_dy):
            return 'G', 0.80

    # H - index middle horizontal
    if index and middle and not ring and not pinky and not thumb:
        if abs(lm[8].x-lm[5].x) > abs(lm[8].y-lm[5].y) * 0.6:
            return 'H', 0.76

    # I - only pinky up
    if pinky and not index and not middle and not ring and not thumb:
        return 'I', 0.92

    # J - pinky + thumb
    if pinky and thumb and not index and not middle and not ring:
        return 'J', 0.82

    # K - index middle spread thumb up
    if index and middle and thumb and not ring and not pinky:
        if spread_im > 0.05:
            return 'K', 0.80

    # L - L shape
    if thumb and index and not middle and not ring and not pinky:
        if index_dy < -0.05 and abs(index_dx) < abs(index_dy):
            return 'L', 0.88

    # M - three fingers over thumb
    if not index and not middle and not ring and not pinky and not thumb:
        if d_ti < 0.12 and d_tm < 0.12 and d_tr < 0.14:
            return 'M', 0.74

    # N - two fingers over thumb
    if not index and not middle and not ring and not pinky and not thumb:
        if d_ti < 0.10 and d_tm < 0.10:
            return 'N', 0.70

    # P - index pointing down
    if index and thumb and not middle and not ring and not pinky:
        if index_dy > 0.05:
            return 'P', 0.78

    # Q - index thumb down
    if index and thumb and not middle and not ring and not pinky:
        if lm[8].y > lm[0].y - 0.05:
            return 'Q', 0.72

    # R - index middle crossed
    if index and middle and not ring and not pinky and not thumb:
        if lm[8].x > lm[12].x + 0.01:
            return 'R', 0.80

    # S - fist thumb over fingers
    if not index and not middle and not ring and not pinky and not thumb:
        if all_curled(lm) and lm[4].y > lm[3].y:
            return 'S', 0.80

    # T - thumb between index and middle
    if not index and not middle and not ring and not pinky and thumb:
        if dist(lm, 4, 6) < 0.07:
            return 'T', 0.76

    # U - index middle up close
    if index and middle and not ring and not pinky and not thumb:
        if spread_im < 0.04:
            return 'U', 0.84

    # V - index middle spread
    if index and middle and not ring and not pinky and not thumb:
        if spread_im >= 0.04:
            return 'V', 0.86

    # W - three fingers up
    if index and middle and ring and not pinky and not thumb:
        return 'W', 0.88

    # X - index hooked
    if not index and not middle and not ring and not pinky and not thumb:
        if lm[8].y - lm[6].y > 0.02:
            return 'X', 0.74

    # Y - thumb pinky shaka
    if thumb and pinky and not index and not middle and not ring:
        return 'Y', 0.92

    # Z - index up alone
    if index and not middle and not ring and not pinky and not thumb:
        return 'Z', 0.72

    return '?', 0.30


def classify_two_hands(lm1, lm2):
    """
    Classify using BOTH hands together.
    Some ISL letters need two hands.
    Returns best guess using combined info.
    """
    # Get single hand results for both
    letter1, conf1 = classify_one_hand(lm1)
    letter2, conf2 = classify_one_hand(lm2)

    # Distance between wrists
    wrist_dist = dist_pts(lm1[0], lm2[0])

    # Distance between index tips of both hands
    index_dist = dist_pts(lm1[8], lm2[8])

    # Both hands forming O shape = special sign
    if is_o_shape(lm1) and is_o_shape(lm2):
        return 'O', 0.92

    # Both hands flat open = B or W or similar
    lm1_index = is_extended(lm1, 8, 6, 5)
    lm1_middle = is_extended(lm1, 12, 10, 9)
    lm1_ring = is_extended(lm1, 16, 14, 13)
    lm1_pinky = is_extended(lm1, 20, 18, 17)

    lm2_index = is_extended(lm2, 8, 6, 5)
    lm2_middle = is_extended(lm2, 12, 10, 9)

    # Both index fingers up close = U
    if lm1_index and lm2_index and not lm1_middle and not lm2_middle:
        if index_dist < 0.10:
            return 'U', 0.82

    # One hand index up + other hand flat = D
    if (lm1_index and not lm1_middle and
        lm2_index and lm2_middle and lm2_ring and lm2_pinky):
        return 'D', 0.80

    # Use whichever hand has higher confidence
    if conf1 >= conf2:
        return letter1, conf1
    else:
        return letter2, conf2


# ── Routes ──

@app.get("/")
def root():
    return {"message": "ISL Translator API v3 - Two Hand Support!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_gesture(req: PredictRequest):
    if len(req.landmarks) != 21:
        raise HTTPException(400, f"Expected 21 landmarks, got {len(req.landmarks)}")

    if req.landmarks2 and len(req.landmarks2) == 21:
        # Two hands detected
        letter, confidence = classify_two_hands(req.landmarks, req.landmarks2)
        return PredictResponse(letter=letter, confidence=confidence, method="two-hand-v3")
    else:
        # One hand
        letter, confidence = classify_one_hand(req.landmarks)
        return PredictResponse(letter=letter, confidence=confidence, method="one-hand-v3")

@app.post("/text-to-sign", response_model=TextToSignResponse)
def text_to_sign(req: TextToSignRequest):
    text = req.text.upper().strip()
    letters, image_urls = [], []
    base_url = "http://localhost:8000/images"
    for char in text:
        if char.isalpha():
            letters.append(char)
            img_path = os.path.join(DATASET_PATH, f"{char}.png")
            image_urls.append(f"{base_url}/{char}.png" if os.path.exists(img_path) else "")
        elif char == " ":
            letters.append(" ")
            image_urls.append("")
    return TextToSignResponse(letters=letters, image_urls=image_urls)

@app.get("/available-signs")
def available_signs():
    return {"signs": [
        {"letter": l, "available": os.path.exists(os.path.join(DATASET_PATH, f"{l}.png"))}
        for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]}