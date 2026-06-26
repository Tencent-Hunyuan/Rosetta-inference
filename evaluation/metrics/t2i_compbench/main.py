import os
import json
import re
import argparse
from datetime import datetime
from PIL import Image

import torch
from torchvision import transforms

from evaluation.metrics.t2i_compbench.metric import T2ICompBenchMetric
from evaluation.datasets.t2i_compbench import T2ICompBenchDataset
from evaluation.constants import VQA_MODEL_PATH

from torch.utils.data import DataLoader

def extract_iter_from_path(img_dir, offset=0):
    """
    从img_dir中提取iter_xxxxx，并可选择性地应用偏移量
    例如: /path/to/samples/iter_0005000/t2i_compbench/images
    如果offset=1000，返回: iter_0006000
    如果offset=0，返回: iter_0005000
    """
    match = re.search(r'iter_(\d+)', img_dir)
    if match:
        iter_num = int(match.group(1))
        new_iter_num = iter_num + offset
        # 保持原有的数字格式（补零）
        original_format = match.group(1)
        padding = len(original_format)
        return f"iter_{new_iter_num:0{padding}d}"
    return None


def extract_base_path(img_dir):
    """
    从img_dir中提取基础路径（到samples之前）
    例如: /path/to/samples/iter_0005000/t2i_compbench/images
    返回: /path/to
    """
    # 找到samples的位置
    samples_idx = img_dir.find('/samples/')
    if samples_idx != -1:
        return img_dir[:samples_idx]
    return None


def check_results_exist(img_dir, output_base_dir=None, offset=0):
    """
    检查目标路径下是否已经存在完整的四个标准JSON文件

    Args:
        img_dir: 图片目录路径
        output_base_dir: 输出目录的基础路径（可选）
        offset: iter的偏移量

    Returns:
        bool: 如果四个文件都存在返回True，否则返回False
    """
    # 提取iter和基础路径（应用偏移量）
    iter_name = extract_iter_from_path(img_dir, offset=offset)
    base_path = extract_base_path(img_dir)

    if not iter_name:
        return False

    # 确定输出目录
    if output_base_dir:
        output_dir = os.path.join(output_base_dir, iter_name, "t2i_compbench", "metric_results")
    elif base_path:
        output_dir = os.path.join(base_path, "samples", iter_name, "t2i_compbench", "metric_results")
    else:
        output_dir = os.path.join(".", iter_name, "t2i_compbench", "metric_results")

    # 检查四个标准JSON文件是否存在
    required_files = [
        't2i_compbench_color.json',
        't2i_compbench_shape.json',
        't2i_compbench_texture.json',
        't2i_compbench_mean.json'
    ]

    for filename in required_files:
        file_path = os.path.join(output_dir, filename)
        if not os.path.exists(file_path):
            return False

    return True


def save_results_to_standard_json(results_dict, img_dir, output_base_dir=None, offset=0):
    """
    将计算结果保存为标准JSON格式

    Args:
        results_dict: metric.compute_metrics() 返回的字典，例如 {'color': (0.466, 300), 'shape': (0.356, 300), ...}
        img_dir: 图片目录路径
        output_base_dir: 输出目录的基础路径（可选）
        offset: iter的偏移量，例如offset=1000时，iter_0005000会变成iter_0006000

    Returns:
        bool: 如果已存在完整结果返回False（跳过），否则返回True（已保存）
    """
    # 检查是否已经存在完整的结果
    if check_results_exist(img_dir, output_base_dir, offset):
        iter_name = extract_iter_from_path(img_dir, offset=offset)
        print(f"跳过保存 {iter_name}: 目标路径下已存在完整的四个标准JSON文件")
        return False

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 提取iter和基础路径（应用偏移量）
    iter_name = extract_iter_from_path(img_dir, offset=offset)
    base_path = extract_base_path(img_dir)

    if not iter_name:
        print(f"无法从img_dir中提取iter: {img_dir}")
        return False

    # 确定输出目录
    if output_base_dir:
        output_dir = os.path.join(output_base_dir, iter_name, "t2i_compbench", "metric_results")
    elif base_path:
        output_dir = os.path.join(base_path, "samples", iter_name, "t2i_compbench", "metric_results")
    else:
        # 如果无法确定路径，使用当前目录
        output_dir = os.path.join(".", iter_name, "t2i_compbench", "metric_results")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个指标
    metrics_to_process = {
        'color': 't2i_compbench_color',
        'shape': 't2i_compbench_shape',
        'texture': 't2i_compbench_texture',
        'avg': 't2i_compbench_mean'
    }

    for metric_key, metric_name in metrics_to_process.items():
        if metric_key in results_dict:
            value, count = results_dict[metric_key]

            # 创建标准格式的JSON数据
            json_data = [{
                "timestamp": timestamp,
                "metric": metric_name,
                "testset": "t2i_compbench",
                "value": value,
                "count": count
            }]

            # 保存到文件
            output_file = os.path.join(output_dir, f"{metric_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

            print(f"已保存: {output_file} (value={value}, count={count})")

    return True


def process_img_dir(img_dir, output_base_dir=None, offset=0,
                     csv_path="evaluation/testsets/test/t2i_compbench.csv",
                     batch_size=64, num_workers=4,
                     skip_if_exists=True):
    """
    处理单个img_dir，计算t2i_compbench指标并保存结果

    Args:
        img_dir: 图片目录路径，例如: /path/to/samples/iter_0005000/t2i_compbench/images
        output_base_dir: 输出目录的基础路径（可选），如果为None则从img_dir中提取
        offset: iter的偏移量，默认为0
        csv_path: t2i_compbench数据集CSV文件路径
        batch_size: 批处理大小
        num_workers: DataLoader的worker数量
        skip_if_exists: 如果结果已存在是否跳过

    Returns:
        bool: 如果成功处理返回True，如果跳过返回False
    """
    # 检查目录是否存在
    if not os.path.exists(img_dir):
        print(f"警告: 目录不存在，跳过: {img_dir}")
        return False

    # 确定输出目录
    if output_base_dir is None:
        output_base_dir = img_dir.split("/iter_")[0]

    print(f"处理图片目录: {img_dir}")
    print(f"输出基础目录: {output_base_dir}")

    # 检查是否已经存在完整的结果
    if skip_if_exists and check_results_exist(
        img_dir=img_dir,
        output_base_dir=output_base_dir,
        offset=offset
    ):
        iter_name = extract_iter_from_path(img_dir, offset=offset)
        print(f"跳过 {iter_name}: 目标路径下已存在完整的四个标准JSON文件，跳过计算")
        return False

    # 初始化metric和dataset
    metric = T2ICompBenchMetric(vqa_model_path=VQA_MODEL_PATH)
    dataset = T2ICompBenchDataset(csvs=csv_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers
    )

    # 处理每个batch
    missing_num = 0
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx} / {len(dataloader)}")

        images = []
        for id in batch["ids"]:
            image_path = os.path.join(img_dir, f"{id}_0.png")
            if not os.path.exists(image_path):
                missing_num += 1
                print(f"警告: 图片不存在: {image_path}")
                continue
            image = Image.open(image_path)
            image = transforms.ToTensor()(image)
            images.append(image)

        if len(images) == 0:
            print(f"警告: batch {batch_idx} 中没有有效图片，跳过")
            continue

        images = torch.stack(images).cuda()

        metric.process(
            images=images,
            questions=batch["questions"],
            dataset_types=batch["dataset_types"],
            prompt=batch["input"],
            ids=batch["ids"]
        )

    print(f"Missing num: {missing_num}")
    results_dict = metric.compute_metrics(metric.results)
    print(f"计算结果: {results_dict}")

    # 保存结果
    save_results_to_standard_json(
        results_dict=results_dict,
        img_dir=img_dir,
        output_base_dir=output_base_dir,
        offset=offset
    )

    return True


def main():
    """主函数，解析命令行参数并处理img_dir"""
    parser = argparse.ArgumentParser(
        description="计算t2i_compbench指标并保存为标准JSON格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个图片目录
  python calculate_t2i_compbench_offline_batch.py --img_dir "/path/to/samples/iter_0005000/t2i_compbench/images"

  # 指定输出目录和偏移量
  python calculate_t2i_compbench_offline_batch.py --img_dir "/path/to/samples/iter_0005000/t2i_compbench/images" --output_base_dir "/output" --offset 1000
        """
    )

    parser.add_argument(
        '--img_dir',
        type=str,
        required=True,
        help='图片目录路径，例如: /path/to/samples/iter_0005000/t2i_compbench/images'
    )

    parser.add_argument(
        '--output_base_dir',
        type=str,
        default=None,
        help='输出目录的基础路径（可选），如果未指定则从img_dir中提取'
    )

    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='iter的偏移量，默认为0'
    )

    parser.add_argument(
        '--csv_path',
        type=str,
        default="evaluation/testsets/test/t2i_compbench.csv",
        help='t2i_compbench CSV path, default: evaluation/testsets/test/t2i_compbench.csv'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='批处理大小，默认为64'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader的worker数量，默认为4'
    )

    parser.add_argument(
        '--no_skip',
        action='store_true',
        help='即使结果已存在也重新计算（默认会跳过已存在的结果）'
    )

    args = parser.parse_args()

    # 处理单个img_dir
    try:
        result = process_img_dir(
            img_dir=args.img_dir,
            output_base_dir=args.output_base_dir,
            offset=args.offset,
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            skip_if_exists=not args.no_skip
        )

        if result:
            print("\n处理成功!")
        else:
            print("\n已跳过（结果已存在）")
    except Exception as e:
        print(f"\n处理失败: {args.img_dir}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
