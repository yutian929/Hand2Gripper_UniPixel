# Modified from https://github.com/DAMO-NLP-SG/VideoRefer/blob/main/videorefer/eval/videorefer_bench_d/1.eval_gpt_new.py

import argparse
import json
import os

import nncore
from tqdm import tqdm

from openai import OpenAI

os.environ['OPENAI_API_KEY'] = '<change-it-to-your-own-key>'


def init():
    client = OpenAI()
    return client


def interaction(client, message_text):
    completion = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    return completion


def main(args):
    client = init()

    paths = nncore.ls(args.input_file, ext=['json', 'jsonl'], join_path=True)
    data = nncore.flatten([nncore.load(path) for path in paths if 'output' in path])
    data = [dict(pred=d['pred'], caption=d['caption']) for d in data]

    with open('unipixel/eval/videorefer_bench_d/system.txt', 'r') as f:
        system_message = f.read()

    total_amount_all = 0
    for d in tqdm(data):
        if 'pred' not in d:
            continue

        gt = '##Correct answer: ' + d['caption'] + '\n'
        pred = '##Predicted answer: ' + d['pred'] + '\n'

        print(gt[:-1])
        print(pred[:-1])

        messages = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_message
            }]
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": gt + pred
            }]
        }]

        for i in range(5):
            try:
                completion = interaction(client, messages)
                generate_content = completion.choices[0].message.content
                d['gpt'] = generate_content

                prompt_tokens_amount = completion.usage.prompt_tokens / 1000000 * 0.15 * 8
                completion_amount = completion.usage.completion_tokens / 1000000 * 0.6 * 8
                # prompt_tokens_amount = completion.usage.prompt_tokens / 1000000 * 2.5 * 8
                # completion_amount = completion.usage.completion_tokens / 1000000 * 10 * 8
                total_amount = prompt_tokens_amount + completion_amount

                text = completion.choices[0].message.content
                assert text, text

                total_amount_all += total_amount

                print(f'Amount: {total_amount} CNY {total_amount / 8} USD')
                print(f'Total Amount: {total_amount_all} CNY {total_amount_all / 8} USD')

                break
            except Exception:
                print("error. model generation failed.")

        b = json.dumps(data)
        f2 = open(args.output_file, 'w')
        f2.write(b)
        f2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()
    client = init()
    main(args)
