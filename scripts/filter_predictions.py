"""
Filter predictions to remove unsupported repositories.
"""

import json
import sys

# Known problematic repos in SWE-bench Pro
UNSUPPORTED_REPOS = {
    'NodeBB/NodeBB',
}

def filter_predictions(input_file, output_file):
    """Filter out predictions for unsupported repos."""

    filtered = []
    skipped = []

    with open(input_file, 'r') as f:
        for line in f:
            pred = json.loads(line.strip())

            # Extract repo from instance_id (format: repo__repo-subname__issue)
            instance_id = pred['instance_id']

            # Check if from unsupported repo
            is_unsupported = any(
                instance_id.startswith(repo.replace('/', '__'))
                for repo in UNSUPPORTED_REPOS
            )

            if is_unsupported:
                skipped.append(instance_id)
            else:
                filtered.append(pred)

    # Write filtered predictions
    with open(output_file, 'w') as f:
        for pred in filtered:
            f.write(json.dumps(pred) + '\n')

    print(f"Filtered predictions:")
    print(f"  Kept: {len(filtered)}")
    print(f"  Skipped: {len(skipped)}")
    if skipped:
        print(f"  Skipped instances: {skipped[:5]}..." if len(skipped) > 5 else f"  Skipped instances: {skipped}")
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_predictions.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    filter_predictions(sys.argv[1], sys.argv[2])
