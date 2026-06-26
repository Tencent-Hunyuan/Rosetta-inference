def load_metric(metric_name, default_max_size=65536, logger=None, **kwargs):
    """
    metric_name pattern: {metric_type}[@{dataset_name}[@{max_size}]]

    Example metric names:
      - fid@coco30k
      - clip_score@coco30k
      - hpsv2@coco30k
      - t2i_compbench
      - mmlu_bench
      - arc_challenge
      - bbh
      - mbpp
      - mmmu
      - mmbench
      - pope
      - ai2d
      - realworldqa
    """
    metric_type, *res = metric_name.split('@')
    dataset_name = res[0] if len(res) > 0 else metric_type
    max_size = int(res[1]) if len(res) > 1 else default_max_size
    if len(res) > 2:
        if logger is None:
            from loguru import logger
        logger.warning(f"Unknown metric arguments: {res[2:]}")

    if metric_type == 'fid':
        from .fid.metric import FIDMetric
        from evaluation.constants import FID_INCEPTION_PATH, FID_TARGET_PATH
        metric = FIDMetric(dims=2048,
                           inception_path=FID_INCEPTION_PATH,
                           target_path=FID_TARGET_PATH[dataset_name],
                           dataset_name=dataset_name,
                           max_size=max_size,
                           )

    elif metric_type == 'clip_score':
        from .clip_score.metric import CLIPScoreMetric
        from evaluation.constants import CLIP_MODEL_PATH
        metric = CLIPScoreMetric(clip_model_path=CLIP_MODEL_PATH,
                                 dataset_name=dataset_name,
                                 max_size=max_size,
                                 )

    elif metric_type == 'hpsv2':
        from .hpsv2.metric import HPSv2Metric
        from evaluation.constants import HPSV2_MODEL_PATH
        metric = HPSv2Metric(hpsv2_model_path=HPSV2_MODEL_PATH,
                             dataset_name=dataset_name,
                             max_size=max_size,
                             )

    elif metric_type == 't2i_compbench':
        from .t2i_compbench.metric import T2ICompBenchMetric
        from evaluation.constants import VQA_MODEL_PATH
        metric = T2ICompBenchMetric(vqa_model_path=VQA_MODEL_PATH,
                                    dataset_name=dataset_name,
                                    max_size=max_size,
                                    )

    elif metric_type == 'mmbench':
        from .MMBench.metric import MMBenchMetric
        from evaluation.constants import LMUDataRoot
        metric = MMBenchMetric(LMUDataRoot=LMUDataRoot,
                               dataset_name=dataset_name)

    elif metric_type == 'mmmu':
        from .MMMU.metric import MMMUMetric
        metric = MMMUMetric(dataset_name=dataset_name)

    elif metric_type == 'pope':
        from .POPE.metric import POPEMetric
        metric = POPEMetric()

    elif metric_type == 'ai2d':
        from .AI2D.metric import AI2DMetric
        metric = AI2DMetric()

    elif metric_type == 'realworldqa':
        from .RealWorldQA.metric import RealWorldQAMetric
        metric = RealWorldQAMetric()

    elif metric_type == 'mmlu_bench':
        from .mmlu.metric import MMLUMetric
        metric = MMLUMetric(dataset_name=dataset_name,
                            prefix_space=kwargs.get('prefix_space', False),
                            **kwargs)

    elif metric_type == 'arc_challenge':
        from .ARC_C.metric import ARCCMetric
        metric = ARCCMetric(dataset_name=dataset_name,
                            prefix_space=kwargs.get('prefix_space', False),
                            **kwargs)

    elif metric_type == 'bbh':
        from .BBH.metric import BBHMetric
        metric = BBHMetric(dataset_name=dataset_name, **kwargs)

    elif metric_type == 'mbpp':
        from .MBPP.metric import MBPPMetric
        metric = MBPPMetric(dataset_name=dataset_name, **kwargs)

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    return metric
