"""Progressively halve flow integration steps or narrow width using coupled-noise teacher targets."""
import torch


def copy_normalization_stats(student, teacher):
    """Copy input/output scale buffers from teacher to student (including narrow students)."""
    for name in ("dx_std", "v_std", "out_dx_std", "out_v_std", "mass_dx_std", "mass_v_std"):
        if hasattr(teacher, name) and hasattr(student, name):
            getattr(student, name).data.copy_(getattr(teacher, name).data)
    if getattr(teacher, "prior", None) is not None and getattr(student, "prior", None) is not None:
        student.prior.rmin.data.copy_(teacher.prior.rmin.data)
    return student


def distillation_loss(student, teacher, pos, vel, types, box, masses,
                      teacher_steps=4, noise=None, supervised_weight=0.0, *,
                      student_steps=None, sampler="euler", charges=None,
                      ff_types=None, bonds=None, stride_fs=None):
    """Match student flow steps (default n/2) to n teacher steps in normalized target units.

    Teacher is evaluated without gradients and its training mode is restored.
    Optional flow-matching supervision uses the teacher endpoint via model.loss.
    Repeat stages with the previous student as a frozen teacher (4 -> 2 -> 1).
    """
    if student_steps is None:
        if teacher_steps < 2 or teacher_steps % 2:
            raise ValueError("teacher_steps must be positive and even when student_steps is omitted")
        student_steps = teacher_steps // 2
    elif student_steps < 1 or teacher_steps < 1:
        raise ValueError("teacher_steps and student_steps must be positive")
    if supervised_weight < 0:
        raise ValueError("supervised_weight must be nonnegative")
    if noise is None:
        noise = teacher.com_free_noise(pos.shape[0], types, masses, like=pos)
    chem = dict(charges=charges, ff_types=ff_types, bonds=bonds, stride_fs=stride_fs)
    training = teacher.training
    try:
        teacher.eval()
        with torch.no_grad():
            dx, vn = teacher.predict(pos, vel, types, box, masses, n_steps=teacher_steps,
                                     noise=noise, sampler=sampler, **chem)
    finally:
        teacher.train(training)
    sx, sv = student.predict_with_grad(pos, vel, types, box, masses,
                                       n_steps=student_steps, noise=noise, sampler=sampler, **chem)
    v_base = student.base_velocity(pos, vel, stride_fs=stride_fs)
    target = student.targets_to_y(dx, vn, v_base, types, masses=masses, stride_fs=stride_fs)
    prediction = student.targets_to_y(sx, sv, v_base, types, masses=masses, stride_fs=stride_fs)
    loss = (prediction - target).square().mean()
    if supervised_weight:
        loss = loss + supervised_weight * student.loss(pos, vel, types, box, masses, dx, vn, **chem)
    return loss
