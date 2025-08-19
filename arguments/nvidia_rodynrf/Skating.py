_base_ = "./default.py"

OptimizationParams = dict(
    densify_grad_threshold = 0.0002,
    densify_grad_threshold_dynamic = 0.00008,
    use_instance_mask=True,
) 