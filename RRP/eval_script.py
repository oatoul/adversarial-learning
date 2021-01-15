import argparse
import sys
if sys.version_info[0] < 3 or sys.version_info[1] < 6:
    raise AssertionError("Please use Python 3.6+")

usage = """

This is the example evaluation script we will be using to calculate your accuracy.
You score will be calculated as the harmonic mean of accuracy in clean and adversarial images, i.e.
score = 2 / (1 / acc_clean + 1 / acc_adversarial)"""

def harmonic_mean(x1, x2):
    eps = 1e-7
    x1 = x1 + eps
    x2 = x2 + eps
    return 2 / (1 / x1 + 1 / x2)

def load_label(path):
    outputs = open(path, 'r').read().splitlines()
    outputs = {i.split('#')[0]: i.split('#')[1] for i in outputs}
    return outputs

p = argparse.ArgumentParser(usage=usage)
required = p.add_argument_group('required arguments')
required.add_argument('--pred-file', type=str, required=True, help='The output of your defense algorithm.')
required.add_argument('--clean-label', type=str, required=False, help='The grounding truth of clean images by CS5260 staff')
required.add_argument('--adv-label', type=str, required=False, help='The grounding truth of adversarial images by CS5260 staff')
args = p.parse_args()

pred        = load_label(args.pred_file)
clean_label = load_label(args.clean_label)
adv_label   = load_label(args.adv_label)
num_pred  = len(pred)
num_clean = len(clean_label)
num_adv   = len(adv_label)

if num_adv + num_clean != num_pred:
    raise AssertionError(f'Number of your predictions {num_pred} does not match the number of labels {num_clean + num_adv}.')

clean_correct = 0
adv_correct = 0
for k, v in pred.items():
    if clean_label.get(k) == v:
        clean_correct += 1
    elif adv_label.get(k) == v:
        adv_correct += 1

score = harmonic_mean(clean_correct / num_clean, adv_correct / num_adv)
print(f"""
Evaluation result:
    Clean:       {clean_correct} / {num_clean} correct.
    Adversarial: {adv_correct} / {num_adv} correct.
    Score:       {score:.4f}.
""")


# python eval_script.py --pred-file=result_RandomPad.txt --clean-label=clean.txt --adv-label=adv.txt
