from app.vision.classifier import frame_confidence


def test_no_detections_yields_zero():
    assert frame_confidence([]) == 0.0


def test_ignores_non_explicit_classes():
    detections = [{"class": "FACE_FEMALE", "score": 0.95}, {"class": "FEET_EXPOSED", "score": 0.9}]
    assert frame_confidence(detections) == 0.0


def test_returns_max_score_among_explicit_classes():
    detections = [
        {"class": "FACE_FEMALE", "score": 0.99},
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.6},
        {"class": "BUTTOCKS_EXPOSED", "score": 0.85},
    ]
    assert frame_confidence(detections) == 0.85
