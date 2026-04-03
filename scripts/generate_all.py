import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import httpx
import json
import time
import os

url = "https://88r1h00efxz4fh-8188.proxy.runpod.net"
client = httpx.Client(timeout=300)
ckpt = "beautifulRealistic_v7.safetensors"

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

# Step 1: Reference face
print("=== Generating reference face ===")
ref_workflow = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 768, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {
        "text": base_prompt + ", close-up portrait, looking at camera, clean background",
        "clip": ["4", 1],
    }},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["4", 1]}},
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 9999, "steps": 35, "cfg": 7.0,
        "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0],
    }},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "mina_ref_v2", "images": ["8", 0]}},
}

resp = client.post(f"{url}/prompt", json={"prompt": ref_workflow})
prompt_id = resp.json()["prompt_id"]
for i in range(120):
    time.sleep(2)
    h = client.get(f"{url}/history/{prompt_id}")
    if prompt_id in h.json():
        outputs = h.json()[prompt_id].get("outputs", {})
        for node_out in outputs.values():
            if "images" in node_out:
                img = node_out["images"][0]
                img_resp = client.get(f"{url}/view", params={
                    "filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": "output"
                })
                with open("assets/mina/references/face_01.png", "wb") as f:
                    f.write(img_resp.content)
                print(f"Reference saved ({len(img_resp.content)//1024}KB)")
                with open("assets/mina/references/face_01.png", "rb") as f:
                    client.post(f"{url}/upload/image",
                                files={"image": ("face_01.png", f, "image/png")},
                                data={"overwrite": "true"})
        break

# Step 2: Get planned items
items_resp = httpx.get("http://localhost:8000/content-queue?persona_id=1&status=planned", timeout=10)
items = items_resp.json()
print(f"\n=== Generating {len(items)} images ===")

success = 0
failed = 0

for item in items:
    item_id = item["id"]
    post_date = item["post_date"]
    concept = item["concept"]
    full_prompt = f"{base_prompt}, wearing stylish clothes, fully clothed, {item['image_prompt']}"

    print(f"\n[{item_id}] {concept} ({post_date})")
    httpx.patch(f"http://localhost:8000/content-queue/{item_id}", json={"status": "generating"}, timeout=10)

    workflow = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 768, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": full_prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["4", 1]}},
        "20": {"class_type": "IPAdapterUnifiedLoaderFaceID", "inputs": {
            "model": ["4", 0], "preset": "FACEID PLUS V2", "lora_strength": 0.85, "provider": "CUDA",
        }},
        "12": {"class_type": "IPAdapterInsightFaceLoader", "inputs": {
            "provider": "CUDA", "model_name": "antelopev2",
        }},
        "13": {"class_type": "LoadImage", "inputs": {"image": "face_01.png"}},
        "10": {"class_type": "IPAdapterFaceID", "inputs": {
            "model": ["20", 0], "ipadapter": ["20", 1], "image": ["13", 0],
            "insightface": ["12", 0],
            "weight": 0.85, "weight_faceidv2": 0.85,
            "weight_type": "linear", "combine_embeds": "concat",
            "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only",
        }},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": hash(post_date + concept + "v2") % (2**32), "steps": 35, "cfg": 7.0,
            "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
            "model": ["10", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0],
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": f"mina_{post_date}", "images": ["8", 0],
        }},
    }

    try:
        resp = client.post(f"{url}/prompt", json={"prompt": workflow})
        if resp.status_code != 200:
            print(f"  Queue error")
            failed += 1
            continue
        prompt_id = resp.json()["prompt_id"]
        for i in range(120):
            time.sleep(2)
            h = client.get(f"{url}/history/{prompt_id}")
            history = h.json()
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    for msg in status.get("messages", []):
                        if msg[0] == "execution_error":
                            print(f"  Error: {msg[1].get('exception_message', 'unknown')[:200]}")
                    failed += 1
                    break
                outputs = history[prompt_id].get("outputs", {})
                for node_out in outputs.values():
                    if "images" in node_out:
                        img = node_out["images"][0]
                        img_resp = client.get(f"{url}/view", params={
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": "output",
                        })
                        out_path = f"assets/mina/generated/{post_date}_{item_id:04d}.png"
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path, "wb") as f:
                            f.write(img_resp.content)
                        print(f"  OK ({len(img_resp.content)//1024}KB)")
                        httpx.patch(
                            f"http://localhost:8000/content-queue/{item_id}",
                            json={"status": "generated"}, timeout=10,
                        )
                        success += 1
                break
        else:
            print(f"  TIMEOUT")
            failed += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        failed += 1

print(f"\n=== DONE: {success} success, {failed} failed ===")
