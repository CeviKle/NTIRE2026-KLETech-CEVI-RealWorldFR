import os.path
import logging
import torch
import argparse
import json
import time
from pprint import pprint
from utils import utils_logger
from utils import utils_image as util


def select_model(args, device):
    model_id = args.model_id
    if model_id == 0:
        from models.team00_CodeFormer import main as CodeFormer
        name = f"{model_id:02}_CodeFormer_baseline"
        model_path = os.path.join('model_zoo', 'team00_CodeFormer')
        model_func = CodeFormer
    elif model_id == 6:
        from models.team06_KLETechCEVI.main import main as model_func
        name = f"{model_id:02}_KLETechCEVI_SAFGRC"
        model_path = os.path.join('model_zoo', 'team06_KLETechCEVI')
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")
    return model_func, model_path, name


def run(model_func, model_name, model_path, device, args, mode="test"):
    if mode == "valid":
        data_path = args.valid_dir
    elif mode == "test":
        data_path = args.test_dir
    assert data_path is not None, "Please specify the dataset path."

    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    # Check if dataset subfolders exist, fallback to flat folder
    dataset_names = ("CelebA", "Wider-Test", "LFW-Test", "WebPhoto-Test", "CelebChild-Test")
    has_subfolders = any(
        os.path.exists(os.path.join(data_path, d)) for d in dataset_names
    )

    if has_subfolders:
        data_paths = []
        save_paths = []
        for dataset_name in dataset_names:
            dp = os.path.join(data_path, dataset_name)
            sp = os.path.join(save_path, dataset_name)
            if os.path.exists(dp):
                data_paths.append(dp)
                save_paths.append(sp)
                util.mkdir(sp)
    else:
        data_paths = [data_path]
        save_paths = [save_path]

    start_time = time.time()

    for dp, sp in zip(data_paths, save_paths):
        try:
            model_func(model_dir=model_path, input_path=dp, output_path=sp, device=device)
        except Exception as e:
            print(f"FATAL ERROR in {dp}: {e}")
            raise e

    elapsed = time.time() - start_time
    print(f"Model {model_name} runtime: {elapsed:.2f} seconds")


def main(args):
    utils_logger.logger_info(
        "NTIRE2026-RealWorld-Face-Restoration",
        log_path="NTIRE2026-RealWorld-Face-Restoration.log"
    )
    logger = logging.getLogger("NTIRE2026-RealWorld-Face-Restoration")

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except:
            pass

    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    model_func, model_path, model_name = select_model(args, device)
    logger.info(model_name)

    if args.valid_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="valid")

    if args.test_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2026-RealWorld-Face-Restoration")
    parser.add_argument("--valid_dir", default=None, type=str)
    parser.add_argument("--test_dir",  default=None, type=str)
    parser.add_argument("--save_dir",  default="NTIRE2026-RealWorld-Face-Restoration/results", type=str)
    parser.add_argument("--model_id",  default=6, type=int)
    args = parser.parse_args()
    pprint(args)
    main(args)
