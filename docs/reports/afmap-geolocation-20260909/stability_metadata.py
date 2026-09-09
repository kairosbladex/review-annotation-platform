"""Lossless Fastjson-compatible references for repeated per-event metadata."""
import copy
import json
import re

def expand_metadata_refs(features):
    result = copy.deepcopy(features)
    def expand(value, visiting):
        if isinstance(value, dict) and "$ref" in value:
            assert set(value) == {"$ref"}
            reference = value["$ref"]
            # Fastjson's generated JSONPath escapes underscores in map keys.
            reference = re.sub(r"\\+_", "_", reference)
            match = re.fullmatch(r"\$\[(\d+)\]\.properties\.geolocation\.([A-Za-z_][A-Za-z_0-9]*)", reference)
            assert match and reference not in visiting, "unsupported or cyclic metadata reference"
            target = result[int(match[1])]["properties"]["geolocation"][match[2]]
            return expand(target, visiting | {reference})
        if isinstance(value, dict): return {k: expand(v, visiting) for k, v in value.items()}
        if isinstance(value, list): return [expand(v, visiting) for v in value]
        return value
    return expand(result, set())

def compact_metadata(features):
    original = expand_metadata_refs(features)
    result = copy.deepcopy(original); seen = {}
    for index, feature in enumerate(result):
        metadata = feature["properties"]["geolocation"]
        for key, value in list(metadata.items()):
            if not isinstance(value, (dict, list)): continue
            signature = json.dumps(value, sort_keys=True, ensure_ascii=False)
            if signature in seen: metadata[key] = {"$ref": seen[signature]}
            else: seen[signature] = f"$[{index}].properties.geolocation.{key}"
    assert expand_metadata_refs(result) == original, "metadata changed during compaction"
    serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    assert len(serialized.encode()) <= 65535, "metadata exceeds current TEXT column"
    return serialized
