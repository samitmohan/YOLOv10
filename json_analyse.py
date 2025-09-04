import os, json, collections

json_root = "/home/samit/samit_workspace/training/odn/data/json"

counter = collections.Counter()
bad_files = 0

for root, dirs, files in os.walk(json_root):
    for fn in files:
        if not fn.endswith(".json"):
            continue
        path = os.path.join(root, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sh in data.get("shapes", []):
                lbl = sh.get("label")
                if lbl:
                    counter[lbl.strip()] += 1
        except Exception as e:
            bad_files += 1
            print("Failed to read:", path, e)

# label analysis
for lbl, count in counter.most_common():
    print(f"{lbl:20s} {count}")

print(f"Total unique labels: {len(counter)}")
if bad_files:
    print(f"Warning: failed to read {bad_files} json files")