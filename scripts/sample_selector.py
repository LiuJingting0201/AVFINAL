import json
import os

# Path config 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "man-truckscenes", "v1.0-mini"))
CONFIGS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "configs"))
OUTPUT_PATH = os.path.join(CONFIGS_DIR, "selected_samples.json")

SCENE_JSON = os.path.join(DATA_DIR, "scene.json")
SAMPLE_JSON = os.path.join(DATA_DIR, "sample.json")

# 2. Target scene tags 
TARGET_SCENES = {
    "scene_night_highway":   ["lighting.dark", "area.highway"],
    "scene_rainy_highway":   ["weather.rain", "area.highway"],
    "scene_overcast_city":   ["weather.overcast", "area.city"],
    "scene_snow_city":       ["weather.snow", "area.city"],
    "scene_terminal_area":   ["area.terminal"],
    "scene_clear_city":      ["weather.clear", "area.city"],
    "scene_bridge_city":     ["structure.bridge", "area.city"],
    "scene_construction":    ["construction.roadworks"],
    "scene_twilight":        ["lighting.twilight"]
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_scene_tokens_by_tags(scenes, required_tags):
    result = []
    for s in scenes:
        tags = s["description"].split(";")
        if all(any(req_tag in tag for tag in tags) for req_tag in required_tags):
            result.append(s)
    return result

def get_sample_tokens_for_scene(scene_token, samples):
    return [s["token"] for s in samples if s["scene_token"] == scene_token]

def analyze_available_tags(scenes):
    all_tags = set()
    for scene in scenes:
        tags = scene["description"].split(";")
        all_tags.update(tags)
    weather_tags = [tag for tag in all_tags if tag.startswith("weather.")]
    area_tags = [tag for tag in all_tags if tag.startswith("area.")]
    lighting_tags = [tag for tag in all_tags if tag.startswith("lighting.")]
    structure_tags = [tag for tag in all_tags if tag.startswith("structure.")]
    construction_tags = [tag for tag in all_tags if tag.startswith("construction.")]
    return {
        "weather": sorted(weather_tags),
        "area": sorted(area_tags),
        "lighting": sorted(lighting_tags),
        "structure": sorted(structure_tags),
        "construction": sorted(construction_tags),
        "all": sorted(all_tags)
    }

def main():
    scenes = load_json(SCENE_JSON)
    samples = load_json(SAMPLE_JSON)
    output = {}

    print("\n=== Tag analysis ===")
    available_tags = analyze_available_tags(scenes)
    for category, tags in available_tags.items():
        if category != "all":
            print(f"{category.upper()}: {tags}")

    print(f"\n=== All scenes and description fields ({len(scenes)} scenes) ===")
    for i, s in enumerate(scenes):
        print(f"{i:2d} | {s['name']}")
        print(f"    desc: {s['description']}")
        print(f"    samples: {s['nbr_samples']}\n")

    print("\n=== Auto tag-based selection ===")
    for tag_name, tag_list in TARGET_SCENES.items():
        selected_scenes = get_scene_tokens_by_tags(scenes, tag_list)
        if not selected_scenes:
            print(f"⚠️  Not found for {tag_list}")
            continue

        output[tag_name] = []
        for scene in selected_scenes:
            print(f"✅ Scene {tag_name}")
            print(f"   Name: {scene['name']}")
            print(f"   Desc: {scene['description']}")
            print(f"   Frames: {scene['nbr_samples']}")
            print()
            sample_tokens = get_sample_tokens_for_scene(scene["token"], samples)
            output[tag_name].extend(sample_tokens[:5])

    print(f"\n=== Selection stats ===")
    total_samples = 0
    for scene_type, sample_list in output.items():
        print(f"{scene_type}: {len(sample_list)} samples")
        total_samples += len(sample_list)
    print(f"Total selected: {total_samples} samples")

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved scene sample list: {OUTPUT_PATH}")

    # Reduce each type to first 2 samples
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    reduced = {scene: tokens[:2] for scene, tokens in all_samples.items()}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(reduced, f, indent=2)
    print("Reduced to 2 samples per type.")

    print("\n[Tip] You can manually edit selected_samples.json to adjust sample count.")
    print("[Tip] For specific frames, filter by timestamp or other criteria if needed.")
    print("[Tip] Default: 2 samples per type, customizable in code.")

if __name__ == "__main__":
    main()
