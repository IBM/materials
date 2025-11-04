import os
import re
import subprocess
import warnings

import yaml


class ConfigParser:
    def __init__(self, args):
        self._args_validation(args)
        self._config_validation()

    def _args_validation(self, args):
        if os.path.exists(args.config):
            with open(args.config, "r") as file:
                self.config = yaml.safe_load(file)
        else:
            raise Exception(f"The path '{args.config}' passed to --config does not exist.")

        if os.path.exists(args.dataset):
            self.config["dataset"]["dataset_path"] = str(args.dataset)
        else:
            raise Exception(f"The path '{args.dataset}' passed to --data does not exist.")

        if not os.path.exists(args.checkpoint):
            raise Exception(f"The path '{args.checkpoint}' passed to --checkpoint does not exist.")
        else:
            self.config["general"]["ckpt_path"] = args.checkpoint

        if args.aim:
            pattern_server = r"^aim://(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5})$"
            pattern_route = r"aim:\/\/[^\\\s]+"
            if self._check_folder_aim(args.aim):
                pass
            elif bool(re.match(pattern_server, args.aim)):
                pass
            elif bool(re.match(pattern_route, args.aim)):
                pass
            else:
                raise Exception(
                    f"The argument '{args.aim}' provided to --aim does not qualify as a folder path ('/xyz'), server path ('aim://IP_ADDRESS:PORT_NUMBER'), nor route URL ('aim://x.y.z')."
                )
            self.config["general"]["aim_path"] = args.aim

        if hasattr(args, "constants") and args.constants is not None:
            if not os.path.exists(args.constants):
                raise Exception(f"The path '{args.constants}'  provided to --constants does not exist.")
            else:
                self.config["dataset"]["constants_path"] = args.constants

        if args.base_model:
            if not os.path.exists(args.base_model):
                raise Exception(f"The path '{args.base_model}'  provided to --base_model does not exist.")
            else:
                self.config["general"]["base_model"] = args.base_model

    def _config_validation(self):
        if self.config["trainer"]["accelerator"] == "cpu":
            if "strategy" in self.config["trainer"]:
                del self.config["trainer"]["strategy"]

            print("\n\nATTENTION: You are training with CPUs only. Multi-GPU Strategy is not necessary for this case.\n\n")

        if (
            self.config["dataset"].get("constants_path") is not None
            and self.config["dataset"].get("max_elements_for_constants") is not None
        ):
            raise Exception(
                "Please choose between providing a path to --constants or 'max_elements_for_constants' in the 'config.yaml' file, but not both."
            )

        if "dataset" in self.config:
            self.config["dataset"]["devices"] = self.config["trainer"]["devices"]
            self.config["dataset"]["seed"] = self.config["general"]["seed"]

        if "encoder" in self.config:
            self.config["encoder"]["args"]["cutoff"] = self.config["dataset"]["cutoff"]

        if "decoder" in self.config:
            self.config["decoder"]["in_channels"] = self.config["encoder"]["args"]["hidden_channels"]
            self.config["decoder"]["num_residues"] = self.config["encoder"]["args"]["num_layers"]

        if not bool(self.config["general"]["debug_mode"]):
            warnings.simplefilter("ignore")

        if "batch_size" in self.config["dataset"].keys() and "max_edges" in self.config["dataset"].keys():
            raise Exception("Both batch_size and max_edges were defined. Please specify only one.")

        if "batch_size" not in self.config["dataset"].keys() and "max_edges" not in self.config["dataset"].keys():
            raise Exception("None of batch_size and max_edges were defined. Please specify exactly one.")

    def __getitem__(self, key):
        return self.config.__getitem__(key)

    def __setitem__(self, key, value):
        self.config[key] = value

    def get(self, key, default=None):
        if key not in self.config:
            self.config[key] = {} if default is None else default
        return self.config[key]

    def _check_folder_aim(self, logger_path):
        if os.path.exists(logger_path):
            return True
        elif logger_path == ".aim":
            try:
                result = subprocess.run(["aim", "init", "--repo", "."], check=True, capture_output=True, text=True)
                print("stdout:", result.stdout)
                return True
            except Exception as e:
                print(str(e))
                return False
        else:
            return False
