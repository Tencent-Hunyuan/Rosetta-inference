import json
import numpy as np
import pandas as pd

from .eval_utils import evaluate, parse_multi_choice_response, parse_open_response, calculate_ins_level_acc
from ..base_metric import BaseMetric
from evaluation.datasets.mmmu import CAT_SHORT2LONG, DOMAIN_CAT2SUB_CAT
from rosetta.utils import safe_file


def calculate_results(results: pd.DataFrame):
    subject = list(CAT_SHORT2LONG.keys())

    # Convert results DataFrame to dict of {'<category>': sub_df}
    results_dict = {}
    for cat_short in subject:
        category = CAT_SHORT2LONG[cat_short]
        sub_df = results[results['category'] == category]
        if len(sub_df) > 0:
            results_dict[category] = sub_df.copy(deep=True)

    # Copied from MMMU repo and modified to avoid save and load.
    evaluation_result = {}
    total_eval_samples = []
    for cat_short in subject:
        category = CAT_SHORT2LONG[cat_short]
        print("Evaluating: {}".format(category))
        if category not in results_dict:
            print("Skipping {} for not found".format(category))
        else:
            cat_outputs = results_dict[category].to_dict('records')
            # Evaluation
            eval_samples = []
            for cat_output in cat_outputs:
                response = cat_output['response']
                if response is None or (isinstance(response, float) and np.isnan(response)):
                    response = ""
                if cat_output['question_type'] == 'multiple-choice':
                    all_choices = cat_output['all_choices']
                    index2ans = cat_output['index2ans']
                    parsed_pred = parse_multi_choice_response(response, all_choices, index2ans)
                    eval_samples.append(
                        {
                            'id': cat_output['id'],
                            'question_type': cat_output['question_type'],
                            'answer': cat_output['answer'],  # the content in option, not answer index.
                            'response': response,
                            'parsed_pred': parsed_pred,
                            'index2ans': index2ans,
                        }
                    )
                else:  # open
                    parsed_pred = parse_open_response(response)
                    eval_samples.append(
                        {
                            'id': cat_output['id'],
                            'question_type': cat_output['question_type'],
                            'answer': cat_output['answer'],
                            'response': response,
                            'parsed_pred': parsed_pred,
                        }
                    )

            print("Num of valid samples: {}, Expected Num: {}".format(len(eval_samples), len(cat_outputs)))

            judge_dict, metric_dict = evaluate(eval_samples)
            metric_dict.update({"num_example": len(eval_samples)})
            for eval_sample in eval_samples:
                eval_sample.update({"judge": judge_dict[eval_sample['id']]})
            total_eval_samples.append(pd.DataFrame(eval_samples))
            evaluation_result[category] = metric_dict

    printable_results = {}
    # pdb.set_trace()
    # add domain Subject
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:  # use the order in DOMAIN_CAT2SUB_CAT
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
        printable_results['Overall-' + domain] = {"num": int(in_domain_data_num),
                                                  "acc": round(in_domain_ins_acc, 3)}
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {"num": int(cat_results['num_example']),
                                           "acc": round(cat_results['acc'], 3)}

    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results['Overall'] = {
        "num": sum([cat_results['num_example'] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 3)
    }

    total_eval_samples = pd.concat(total_eval_samples)
    return total_eval_samples, printable_results


class MMMUMetric(BaseMetric):
    def __init__(self, dataset_name="mmmu"):
        super().__init__()
        self.dataset_name = dataset_name
        self.results = []
        # {} for timestamp
        self.save_file_template = f"{dataset_name}_{{}}.csv"

    def load_model(self, logger=None):
        if logger is not None:
            logger.info("MMMUMetric: loaded.")

    def release_model(self):
        pass

    def process(self, answers, lines, **kwargs):
        lines["response"] = answers
        self.results.append(lines)

    def compute_metrics(self, results, save_file=None):
        df = pd.concat(results, ignore_index=True)

        total_eval_samples, printable_results = calculate_results(df)
        if save_file is not None:
            total_eval_samples.to_csv(safe_file(save_file), index=False)

        print(json.dumps(printable_results, indent=4, ensure_ascii=False))

        return float(printable_results['Overall']['acc']), int(printable_results['Overall']['num'])
