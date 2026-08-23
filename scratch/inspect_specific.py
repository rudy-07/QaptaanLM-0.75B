import sys
import json
import time
from datasets import load_dataset

def test_stream_first(repo_id, config=None, split='train', data_files=None):
    print(f"\n==========================================")
    print(f"Inspecting: {repo_id} (cfg={config}, split={split})")
    try:
        kw = {'split': split, 'streaming': True}
        if config: kw['name'] = config
        if data_files: kw['data_files'] = data_files
        ds = load_dataset(repo_id, **kw)
        row = next(iter(ds))
        print(f"SUCCESS! Keys: {list(row.keys())}")
        for k in list(row.keys())[:6]:
            val = row[k]
            if isinstance(val, list):
                print(f"  {k}: list[{len(val)}] -> {repr(val[:1])[:100]}")
            elif isinstance(val, dict):
                print(f"  {k}: dict keys={list(val.keys())}")
            else:
                print(f"  {k}: {repr(val)[:100]}")
        return row
    except Exception as e:
        print(f"FAILED: {e}")
        return None

if __name__ == '__main__':
    test_stream_first('allenai/tulu-3-sft-mixture', split='train')
    test_stream_first('bigcode/self-oss-instruct-sc2-exec-filter-50k', split='train')
    test_stream_first('teknium/OpenHermes-2.5', split='train')
    test_stream_first('open-thoughts/OpenThoughts-114k', split='train')
    test_stream_first('HuggingFaceTB/smoltalk', 'smol-constraints', split='train')
    test_stream_first('m-a-p/CodeFeedback-Filtered-Instruction', split='train')
    test_stream_first('allenai/sciq', split='train')
    test_stream_first('theblackcat102/evol-codealpaca-v1', split='train')
