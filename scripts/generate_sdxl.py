import httpx
import time
import os

url = "https://88r1h00efxz4fh-8188.proxy.runpod.net"
client = httpx.Client(timeout=300)

base_prompt = (
    "young korean woman, 23 years old, slim build, "
    "long straight black hair with natural volume and center-to-side part, "
    "oval face with slim V-line jaw, bright clear skin with fair luminous complexion, "
    "large expressive double-eyelid eyes with soft gentle gaze, "
    "natural peach-coral lips, subtle natural makeup with dewy finish, "
    "warm lovely radiant smile, clean elegant impression, "
    "8k uhd, natural skin texture, soft studio lighting, "
    "photorealistic, DSLR quality, shallow depth of field"
)
neg_prompt = (
    "deformed, blurry, cartoon, anime, 3d render, painting, "
    "low quality, bad anatomy, extra fingers, mutated hands, "
    "watermark, text, signature, nude, naked, nsfw, "
    "revealing clothing, cleavage, lingerie, underwear, bare skin, topless"
)

MODELS = [
    ("realvisxl_v50.safetensors", "realvis"),
    ("juggernautXL_v10.safetensors", "jugger"),
]


def generate_ref(ckpt, prefix):
    """Generate reference face with given checkpoint."""
    workflow = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {
            "text": base_prompt + ", close-up portrait, looking at camera, clean background, wearing white blouse",
            "clip": ["4", 1],
        }},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": 9999, "steps": 30, "cfg": 7.0,
            "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0],
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"mina_ref_{prefix}", "images": ["8", 0]}},
    }
    resp = client.post(f"{url}/prompt", json={"prompt": workflow})
    prompt_id = resp.json()["prompt_id"]
    for _ in range(120):
        time.sleep(2)
        h = client.get(f"{url}/history/{prompt_id}").json()
        if prompt_id in h:
            for node_out in h[prompt_id].get("outputs", {}).values():
                if "images" in node_out:
                    img = node_out["images"][0]
                    img_resp = client.get(f"{url}/view", params={
                        "filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": "output",
                    })
                    ref_path = f"assets/mina/references/face_{prefix}.png"
                    with open(ref_path, "wb") as f:
                        f.write(img_resp.content)
                    # Upload to ComfyUI
                    with open(ref_path, "rb") as f:
                        client.post(f"{url}/upload/image",
                                    files={"image": (f"face_{prefix}.png", f, "image/png")},
                                    data={"overwrite": "true"})
                    print(f"  Reference saved: {ref_path} ({len(img_resp.content)//1024}KB)")
                    return f"face_{prefix}.png"
    print("  TIMEOUT generating reference")
    return None


def generate_image(ckpt, prefix, ref_name, prompt_text, post_date, item_id):
    """Generate one image with FaceID using SDXL model."""
    full_prompt = f"{base_prompt}, wearing stylish clothes, fully clothed, {prompt_text}"

    workflow = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": full_prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["4", 1]}},
        "20": {"class_type": "IPAdapterUnifiedLoaderFaceID", "inputs": {
            "model": ["4", 0], "preset": "FACEID PORTRAIT UNNORM - SDXL only (strong)",
            "lora_strength": 0.7, "provider": "CUDA",
        }},
        "12": {"class_type": "IPAdapterInsightFaceLoader", "inputs": {
            "provider": "CUDA", "model_name": "antelopev2",
        }},
        "13": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "10": {"class_type": "IPAdapterFaceID", "inputs": {
            "model": ["20", 0], "ipadapter": ["20", 1], "image": ["13", 0],
            "insightface": ["12", 0],
            "weight": 0.8, "weight_faceidv2": 0.8,
            "weight_type": "linear", "combine_embeds": "concat",
            "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only",
        }},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": hash(post_date + prefix) % (2**32), "steps": 30, "cfg": 7.0,
            "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
            "model": ["10", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0],
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": f"mina_{prefix}_{post_date}", "images": ["8", 0],
        }},
    }

    resp = client.post(f"{url}/prompt", json={"prompt": workflow})
    if resp.status_code != 200:
        print(f"  Queue error: {resp.text[:150]}")
        return False
    prompt_id = resp.json()["prompt_id"]
    for i in range(120):
        time.sleep(2)
        h = client.get(f"{url}/history/{prompt_id}").json()
        if prompt_id in h:
            status = h[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                for msg in status.get("messages", []):
                    if msg[0] == "execution_error":
                        print(f"  Error: {msg[1].get('exception_message', '')[:100]}")
                return False
            for node_out in h[prompt_id].get("outputs", {}).values():
                if "images" in node_out:
                    img = node_out["images"][0]
                    img_resp = client.get(f"{url}/view", params={
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": "output",
                    })
                    out_path = f"assets/mina/generated/{prefix}/{post_date}_{item_id:04d}.png"
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(img_resp.content)
                    print(f"  OK ({len(img_resp.content)//1024}KB)")
                    return True
            break
    print("  TIMEOUT")
    return False


# Get content items
items_resp = httpx.get("http://localhost:8000/content-queue?persona_id=1&status=planned", timeout=10)
items = items_resp.json()
if not items:
    # Also try generating status
    items_resp = httpx.get("http://localhost:8000/content-queue?persona_id=1", timeout=10)
    all_items = items_resp.json()
    items = [i for i in all_items if i["post_date"] >= "2026-04-03"][-7:]
print(f"Using {len(items)} content items\n")

for ckpt, prefix in MODELS:
    print(f"\n{'='*50}")
    print(f"MODEL: {ckpt}")
    print(f"{'='*50}")

    print("\nStep 1: Reference face")
    ref_name = generate_ref(ckpt, prefix)
    if not ref_name:
        continue

    print(f"\nStep 2: Generating {len(items)} images")
    success = 0
    failed = 0
    for item in items:
        item_id = item["id"]
        post_date = item["post_date"]
        concept = item["concept"]
        print(f"\n[{item_id}] {concept} ({post_date})")
        if generate_image(ckpt, prefix, ref_name, item["image_prompt"], post_date, item_id):
            success += 1
        else:
            failed += 1
    print(f"\n{prefix}: {success} success, {failed} failed")

print("\n=== ALL DONE ===")
