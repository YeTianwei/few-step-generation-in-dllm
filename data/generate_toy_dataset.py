"""Generate a toy VLA dataset with LLM APIs.

Run:
    conda activate ~/miniconda3/envs/dllm
    set GEMINI_API_KEY=your_api_key
    python d:\master\HKUST RA\BlockDiff-VLA\dllm\data\generate_toy_dataset.py
"""

import asyncio
import json
import os
import random
import re
from google import genai
from google.genai import types

# ================= 配置区 =================
API_KEY_ENV_VAR = "GEMINI_API_KEY"
MODEL_NAME = "gemini-2.5-flash"  # 在当前环境中已验证可用

TARGET_TOTAL_SAMPLES = 2000
SAMPLES_PER_REQUEST = 10         # 先用更小批次提高稳定性
CONCURRENCY_LIMIT = 1            # 先单并发跑通，再逐步加速
OUTPUT_FILE = "toy_vla_dataset.jsonl"
REQUEST_TIMEOUT = 60             # 设置 60 秒超时限制
MAX_IDLE_BATCHES = 5             # 连续若干批次没有新增样本就停止
MAX_OUTPUT_TOKENS = 2048         # 控制单次输出长度，避免长响应超时
# ==========================================

SYSTEM_INSTRUCTION = """
You are an expert in robotics data synthesis.
Return only valid JSON that matches the provided schema.
Each sample must describe the same high-level task while varying wording naturally.
Do not wrap the JSON in markdown fences.
"""

INSTRUCTION_MARKER = "Instruction:"
TEXT_START_MARKER = "Assistant response:"
TEXT_END_MARKER = "Action sequence:"
ACTION_END_MARKER = "End of plan."
RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "instruction": {"type": "STRING"},
            "assistant_response": {"type": "STRING"},
            "action_sequence": {"type": "STRING"},
        },
        "required": ["instruction", "assistant_response", "action_sequence"],
    },
}

# Inspired by common tabletop manipulation tasks from LIBERO / CALVIN:
# pick-and-place, drawer/cabinet interaction, stacking, pushing, and
# relative placement under simple household scenes.
SCENARIOS = [
    "pick up the red block and place it in the blue bowl",
    "pick up the yellow block and place it in the green bowl",
    "pick up the blue block and place it in the wooden tray",
    "pick up the pink cube and place it on the white plate",
    "pick up the orange cube and place it inside the black bin",
    "move the red block to the left of the blue block",
    "move the yellow block to the right of the green block",
    "place the blue cube in front of the red cube",
    "place the green cube behind the yellow cube",
    "arrange the red, blue, and green blocks into a straight line",
    "stack the red block on top of the blue block",
    "stack the yellow block on top of the green block",
    "stack the blue cube on the red cube and place the stack on the plate",
    "build a two-block tower with the green block at the bottom",
    "unstack the top block and place it in the bowl",
    "push the red block to the target zone",
    "push the blue block next to the yellow block",
    "slide the green block into the tray",
    "slide the pink cube away from the edge of the table",
    "nudge the orange block so it is centered on the coaster",
    "open the drawer",
    "open the drawer and place the red block inside",
    "open the drawer and place the spoon inside",
    "open the drawer, move the blue block into it, and close the drawer",
    "pull the drawer halfway open and inspect the object inside",
    "close the open drawer",
    "take the block out of the drawer and place it on the table",
    "take the spoon out of the drawer and place it in the bowl",
    "open the cabinet door",
    "open the cabinet door and place the cup inside",
    "open the cabinet, put the plate inside, and close the door",
    "close the cabinet door after retrieving the bowl",
    "take the mug out of the cabinet and place it on the coaster",
    "pick up the bottle and place it inside the cabinet",
    "move the can from the cabinet to the tray",
    "place the lid on the pot",
    "remove the lid from the pot and place it on the table",
    "pick up the cup and place it on the coaster",
    "move the mug next to the kettle",
    "place the bottle behind the cup",
    "move the can to the right of the bottle",
    "put the bowl on the plate",
    "place the spoon into the cup",
    "place the fork next to the plate",
    "move the knife onto the cutting board",
    "put the carrot in the pot",
    "place the potato into the pan",
    "move the apple into the fruit bowl",
    "place the banana next to the apple",
    "pick up the cucumber and place it on the cutting board",
    "move the tomato from the plate to the bowl",
    "sort the red block and blue block into different containers",
    "sort the spoon, fork, and knife into the tray slots",
    "group the cup and bowl together on the left side of the table",
    "separate the stacked blocks and place them in two bowls",
    "gather all loose objects into the tray",
    "clear the table by putting every object into the bin",
    "place the red block on the coaster and the blue block in the bowl",
    "put the spoon in the drawer and the cup on the plate",
    "move the apple into the bowl and the banana onto the plate",
    "open the drawer, take out the block, and stack it on the cube",
    "open the cabinet, retrieve the mug, and place it next to the kettle",
    "pick up the cloth and wipe the area in front of the plate",
    "move the sponge next to the sink",
    "place the brush inside the container",
    "put the towel on the rack",
    "move the soap bottle to the back corner of the counter",
    "pick up the toy car and place it in the garage tray",
    "move the toy animal next to the small house",
    "stack the two rings on the peg",
    "place the cylinder into the matching hole",
    "move the triangle block onto the board",
    "put the red block into the top-left corner of the tray",
    "put the blue block into the bottom-right corner of the tray",
    "place the green cube at the center of the mat",
    "move the mug to the far-left side of the table",
    "move the bowl to the near edge of the counter",
    "reposition the bottle to the back-right corner",
    "place the spoon above the plate",
    "place the fork below the bowl",
    "move the knife to the left of the cutting board",
    "put the cup between the bowl and the plate",
    "place the bottle between the mug and the can",
    "move the red block closer to the yellow block",
    "move the blue cube farther away from the green cube",
    "align the two cups side by side",
    "align the three blocks by color order from left to right",
    "rotate the bottle upright and place it on the coaster",
    "stand the fallen cup upright",
    "turn the bowl upside down and place it on the plate",
    "return the drawer to the fully closed position",
    "reopen the cabinet door and place the can on the lower shelf",
    "move the plate from the cabinet to the table",
    "transfer the block from the bowl into the drawer",
    "transfer the spoon from the drawer to the cup",
    "transfer the mug from the table into the cabinet",
    "take the block off the plate and place it in the bin",
    "remove the apple from the bowl and place it on the cutting board",
    "move the carrot from the cutting board into the pot",
    "put the lid back on the pot after placing the carrot inside",
    "tidy the workspace by placing every utensil into the drawer",
    "prepare the table by placing the plate, cup, and spoon neatly together",
    "set up a simple breakfast scene with a bowl, spoon, and banana",
    "reset the scene by returning all blocks to the tray",
]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
    if text.endswith("```"):
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _clean_field(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_and_validate_response(content: str) -> list[dict[str, str]]:
    """Parse LLM output into strict proxy-format samples."""
    if not content or not content.strip():
        return []

    normalized = _strip_code_fences(content)
    sample_pattern = re.compile(
        rf"(?:^|\n)\s*(?:[-*]\s*|\d+[.)]\s*)?{re.escape(INSTRUCTION_MARKER)}\s*(?P<instruction>.*?)"
        rf"\s*{re.escape(TEXT_START_MARKER)}\s*(?P<text_target>.*?)"
        rf"\s*{re.escape(TEXT_END_MARKER)}\s*(?P<action_target>.*?)"
        rf"\s*{re.escape(ACTION_END_MARKER)}(?=\s*(?:\n\s*(?:[-*]\s*|\d+[.)]\s*)?{re.escape(INSTRUCTION_MARKER)}|\Z))",
        re.DOTALL,
    )

    parsed_samples: list[dict[str, str]] = []
    seen = set()

    for match in sample_pattern.finditer(normalized):
        instruction = _clean_field(match.group("instruction"))
        text_target = _clean_field(match.group("text_target"))
        action_target = _clean_field(match.group("action_target"))

        if not instruction or not text_target or not action_target:
            continue

        fields = (instruction, text_target, action_target)
        if any(
            marker in field
            for field in fields
            for marker in (
                INSTRUCTION_MARKER,
                TEXT_START_MARKER,
                TEXT_END_MARKER,
                ACTION_END_MARKER,
            )
        ):
            continue

        if fields in seen:
            continue
        seen.add(fields)

        parsed_samples.append(
            {
                "instruction": instruction,
                "text_target": text_target,
                "action_target": action_target,
                "raw_text": (
                    f"{INSTRUCTION_MARKER} {instruction}\n"
                    f"{TEXT_START_MARKER} {text_target} {TEXT_END_MARKER}\n"
                    f"{TEXT_END_MARKER} {action_target} {ACTION_END_MARKER}"
                ),
            }
        )

    return parsed_samples


def parse_and_validate_json_response(content: str) -> list[dict[str, str]]:
    """Parse structured JSON output and convert it to proxy-format samples."""
    if not content or not content.strip():
        return []

    try:
        payload = json.loads(_strip_code_fences(content))
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []

    parsed_samples: list[dict[str, str]] = []
    seen = set()
    for item in payload:
        if not isinstance(item, dict):
            continue

        instruction = _clean_field(str(item.get("instruction", "")))
        text_target = _clean_field(str(item.get("assistant_response", "")))
        action_target = _clean_field(str(item.get("action_sequence", "")))

        if not instruction or not text_target or not action_target:
            continue

        fields = (instruction, text_target, action_target)
        if any(
            marker in field
            for field in fields
            for marker in (
                INSTRUCTION_MARKER,
                TEXT_START_MARKER,
                TEXT_END_MARKER,
                ACTION_END_MARKER,
            )
        ):
            continue

        if fields in seen:
            continue
        seen.add(fields)

        parsed_samples.append(
            {
                "instruction": instruction,
                "text_target": text_target,
                "action_target": action_target,
                "raw_text": (
                    f"{INSTRUCTION_MARKER} {instruction}\n"
                    f"{TEXT_START_MARKER} {text_target} {TEXT_END_MARKER}\n"
                    f"{TEXT_END_MARKER} {action_target} {ACTION_END_MARKER}"
                ),
            }
        )

    return parsed_samples

def create_genai_client() -> genai.Client:
    """Create a Gemini client using the stable sync transport."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"Missing {API_KEY_ENV_VAR}. Set it in your environment before running this script."
        )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
    )


async def fetch_samples(client, semaphore, pbar_dict):
    scenario = random.choice(SCENARIOS)
    prompt = (
        f"Generate exactly {SAMPLES_PER_REQUEST} diverse robotics samples for this "
        f"scenario: {scenario}\n"
        "Requirements:\n"
        "- Return a JSON array only.\n"
        f"- The array must contain exactly {SAMPLES_PER_REQUEST} objects.\n"
        "- Each object must have keys: instruction, assistant_response, action_sequence.\n"
        "- instruction: one natural-language user command.\n"
        "- assistant_response: concise reasoning or plan summary in one sentence.\n"
        "- action_sequence: a concise token-like action plan as plain text.\n"
        "- Keep all samples logically consistent with the scenario.\n"
        "- Do not include markdown, comments, or any extra text."
    )
    
    async with semaphore:
        for attempt in range(3):
            try:
                print(f"📡 正在发送请求 (尝试 {attempt+1})...")
                # 使用新版 SDK 的异步调用
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.4,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                    ),
                )
                
                content = response.text
                samples = parse_and_validate_json_response(content)
                if not samples:
                    samples = parse_and_validate_response(content)
                
                if samples:
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                        for s in samples:
                            f.write(json.dumps(s, ensure_ascii=False) + '\n')
                    pbar_dict['generated'] += len(samples)
                    print(f"✅ 成功! 获得 {len(samples)} 条 | 总数: {pbar_dict['generated']}/{TARGET_TOTAL_SAMPLES}")
                    return
                else:
                    print("⚠️ 请求成功，但返回内容未通过格式校验。")
                    
            except Exception as e:
                print(f"❌ 错误 [{type(e).__name__}]: {str(e)[:200]}...")
                # 遇到频率限制 (429) 或其他错误时，等待时间翻倍
                await asyncio.sleep(5 * (attempt + 1))

        print("⚠️ 当前任务在 3 次尝试后仍失败，跳过这一批。")
                
async def main():
    # 初始化输出文件
    with open(OUTPUT_FILE, 'w') as f: pass

    client = create_genai_client()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    pbar_dict = {'generated': 0}
    idle_batches = 0
    
    print(f"🚀 启动 V2 版脚本 | 目标: {TARGET_TOTAL_SAMPLES} | 模型: {MODEL_NAME}")

    try:
        while pbar_dict['generated'] < TARGET_TOTAL_SAMPLES:
            generated_before = pbar_dict['generated']
            tasks = [
                fetch_samples(client, semaphore, pbar_dict)
                for _ in range(CONCURRENCY_LIMIT)
            ]
            await asyncio.gather(*tasks)

            if pbar_dict['generated'] == generated_before:
                idle_batches += 1
                print(
                    f"⚠️ 本轮没有新增样本，连续空转批次: "
                    f"{idle_batches}/{MAX_IDLE_BATCHES}"
                )
            else:
                idle_batches = 0

            if idle_batches >= MAX_IDLE_BATCHES:
                print("🛑 连续多轮没有进展，主动停止。请检查网络、模型名或输出格式。")
                break

            # 每个并发批次后强制喘息，防止触发频率限制
            await asyncio.sleep(2)
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(main())
