def fuse_scores(video_score: float | None, audio_score: float | None, alpha: float = 0.6) -> float:
    """
    Combine video+audio into a single fake probability.
    If one is None, return the other.
    """
    if video_score is None and audio_score is None:
        return 0.5
    if video_score is None:
        return audio_score
    if audio_score is None:
        return video_score
    return float(alpha * video_score + (1 - alpha) * audio_score)

def trust_score(fake_prob: float) -> float:
    """
    Convert fake probability to 'Trust Score' (100% real -> high score)
    """
    return float((1.0 - fake_prob) * 100.0)
