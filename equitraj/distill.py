"""Progressively halve flow integration steps using coupled-noise teacher targets."""
import torch


def distillation_loss(student, teacher, pos, vel, types, box, masses,
                      teacher_steps=10, noise=None, supervised_weight=0.0):
    """Match n/2 student steps to n teacher steps in normalized target units.

    Teacher is evaluated without gradients and its training mode is restored.
    Optional flow-matching supervision uses the teacher endpoint via model.loss.
    Repeat stages with the previous student as a frozen teacher (4→2→1).
    """
    if teacher_steps < 2 or teacher_steps % 2:
        raise ValueError("teacher_steps must be positive and even")
    if supervised_weight < 0:
        raise ValueError("supervised_weight must be nonnegative")
    if noise is None:
        noise = teacher.com_free_noise(pos.shape[0], types, masses, like=pos)
    training = teacher.training
    try:
        teacher.eval()
        with torch.no_grad():
            dx, vn = teacher.predict(pos, vel, types, box, masses, n_steps=teacher_steps, noise=noise)
    finally:
        teacher.train(training)
    sx, sv = student.predict_with_grad(pos, vel, types, box, masses,
                                       n_steps=teacher_steps // 2, noise=noise)
    target = student.targets_to_y(dx, vn, vel[:, -1], types)
    prediction = student.targets_to_y(sx, sv, vel[:, -1], types)
    loss = (prediction - target).square().mean()
    if supervised_weight:
        loss = loss + supervised_weight * student.loss(pos, vel, types, box, masses, dx, vn)
    return loss
