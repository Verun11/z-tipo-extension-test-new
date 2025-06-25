import scripts

print(scripts, scripts.__file__, dir(scripts))

import os
import json
import pathlib
import random
from functools import lru_cache

import torch
import gradio as gr

import modules.scripts as scripts
from modules import devices, shared, options
from modules.scripts import basedir, OnComponent
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)
from modules.prompt_parser import parse_prompt_attention
from modules.extra_networks import parse_prompt
from modules.shared import opts

if hasattr(opts, "hypertile_enable_unet"):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.executor.tipo import (
    parse_tipo_request,
    tipo_runner,
    apply_tipo_prompt,
    parse_tipo_result,
)
from kgen.formatter import seperate_tags, apply_format
from kgen.metainfo import TARGET, TIPO_DEFAULT_FORMAT
from kgen.logging import logger


ext_dir = basedir()
models.model_dir = pathlib.Path(ext_dir) / "models"


SEED_MAX = 2**31 - 1
QUOTESWAP = str.maketrans("'\"", "\"'")
TOTAL_TAG_LENGTH = {
    "VERY_SHORT": "very short",
    "SHORT": "short",
    "LONG": "long",
    "VERY_LONG": "very long",
}
PROCESSING_TIMING = {
    "BEFORE": "Before applying other prompt processings",
    "AFTER": "After applying other prompt processings",
}
DEFAULT_FORMAT = """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>, 

<|quality|>, <|meta|>, <|rating|>"""
TIMING_INFO_TEMPLATE = (
    "_Prompt upsampling will be applied to {} "
    "sd-dynamic-promps and the webui's styles feature are applied_"
)
INFOTEXT_KEY = "TIPO Parameters"
INFOTEXT_KEY_PROMPT = "TIPO prompt"
INFOTEXT_NL_PROMPT = "TIPO nl prompt"
INFOTEXT_KEY_FORMAT = "TIPO format"

PROMPT_INDICATE_HTML = """
<div style="height: 100%; width: 100%; display: flex; justify-content: center; align-items: center">
    <span>
        Original Prompt Loaded.<br>
        Click "Apply" to apply the original prompt.
    </span>
</div>
"""
RECOMMEND_MARKDOWN = """
### Recommended Model and Settings:

"""
MODEL_NAME_LIST = [
    f"{model_name} | {file}"
    for model_name, ggufs in models.tipo_model_list
    for file in ggufs
] + [i[0] for i in models.tipo_model_list]


def on_process_timing_dropdown_changed(timing: str):
    info = ""
    if timing == PROCESSING_TIMING["BEFORE"]:
        info = "**only the first image in batch**, **before**"
    elif timing == PROCESSING_TIMING["AFTER"]:
        info = "**all images in batch**, **after**"
    else:
        raise ValueError(f"Unknown timing: {timing}")
    return TIMING_INFO_TEMPLATE.format(info)


def apply_strength(tag_map, strength_map, strength_map_nl, break_map):
    for cate in tag_map.keys():
        new_list = []
        # Skip natural language output at first
        if isinstance(tag_map[cate], str):
            # Ensure all the parts in the strength_map are in the prompt
            if all(part in tag_map[cate] for part, strength in strength_map_nl):
                org_prompt = tag_map[cate]
                new_prompt = ""
                for part, strength in strength_map_nl:
                    before, org_prompt = org_prompt.split(part, 1)
                    new_prompt += before.replace("(", "\(").replace(")", "\)")
                    part = part.replace("(", "\(").replace(")", "\)")
                    new_prompt += f"({part}:{strength})"
                new_prompt += org_prompt
            tag_map[cate] = new_prompt
            continue
        for org_tag in tag_map[cate]:
            tag = org_tag.replace("(", "\(").replace(")", "\)")
            if org_tag in strength_map:
                new_list.append(f"({tag}:{strength_map[org_tag]})")
            else:
                new_list.append(tag)
            if tag in break_map or org_tag in break_map:
                new_list.append("BREAK")
        tag_map[cate] = new_list

    return tag_map


class TIPOScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.prompt_area = [None, None, None, None]
        self.tag_prompt_area = [None, None]
        self.prompt_area_row = [None, None]
        self.current_model = None
        self.on_after_component_elem_id = [
            ("txt2img_prompt_row", lambda x: self.create_new_prompt_area(0, x)),
            ("txt2img_prompt", lambda x: self.set_prompt_area(0, x)),
            ("img2img_prompt_row", lambda x: self.create_new_prompt_area(1, x)),
            ("img2img_prompt", lambda x: self.set_prompt_area(1, x)),
        ]

    def create_new_prompt_area(self, i2i: int, prompt_row: OnComponent):
        with prompt_row.component:
            with gr.Column(visible=not opts.tipo_no_extra_input):
                new_tag_prompt_area = gr.Textbox(
                    label="Tag Prompt",
                    lines=3,
                    placeholder="Tag Prompt for TIPO (Put Tags to Prompt region)",
                )
                new_prompt_area = gr.Textbox(
                    label="Natural Language Prompt",
                    lines=3,
                    placeholder="Natural Language Prompt for TIPO (Put Tags to Prompt region)",
                )
        self.tag_prompt_area[i2i] = new_tag_prompt_area
        self.prompt_area_row[i2i] = gr.Row()
        # with self.prompt_area_row[i2i]:
        self.prompt_area[i2i * 2 + 1] = new_prompt_area

    def set_prompt_area(self, i2i: int, component: OnComponent):
        self.prompt_area[i2i * 2] = component.component

    def title(self):
        return "TIPO"

    def show(self, _):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with self.prompt_area_row[is_img2img]:
            with gr.Column(
                scale=1, min_width=180, visible=not opts.tipo_no_extra_input
            ):
                prompt_gen = gr.Button(value="Generate Prompt")
            with gr.Column(scale=6):
                with (
                    InputAccordion(False, open=False, label=self.title())
                    if InputAccordion
                    else gr.Accordion(open=False, label=self.title())
                ) as tipo_acc:
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(scale=1):
                                    if InputAccordion is None:
                                        enabled_check = gr.Checkbox(
                                            label="Enabled", value=False, min_width=20
                                        )
                                    else:
                                        enabled_check = tipo_acc
                                    read_orig_prompt_btn = gr.Button(
                                        size="sm",
                                        value="Apply original prompt",
                                        visible=False,
                                        min_width=20,
                                    )
                                with gr.Column(scale=3):
                                    orig_prompt_area = gr.TextArea(visible=False)
                                    orig_prompt_light = gr.HTML("")
                                orig_prompt_area.change(
                                    lambda x: PROMPT_INDICATE_HTML * bool(x),
                                    inputs=orig_prompt_area,
                                    outputs=orig_prompt_light,
                                )
                                orig_prompt_area.change(
                                    lambda x: gr.update(visible=bool(x)),
                                    inputs=orig_prompt_area,
                                    outputs=read_orig_prompt_btn,
                                )
                                read_orig_prompt_btn.click(
                                    fn=lambda x: x,
                                    inputs=[orig_prompt_area],
                                    outputs=self.prompt_area[is_img2img],
                                )

                            tag_length_radio = gr.Radio(
                                label="Tags Length target",
                                choices=list(TOTAL_TAG_LENGTH.values()),
                                value=TOTAL_TAG_LENGTH["LONG"],
                            )
                            nl_length_radio = gr.Radio(
                                label="NL Length target",
                                choices=list(TOTAL_TAG_LENGTH.values()),
                                value=TOTAL_TAG_LENGTH["LONG"],
                            )
                            ban_tags_textbox = gr.Textbox(
                                label="Ban tags",
                                info="Separate with comma. Regex supported.",
                                value="",
                                placeholder="umbrella, official.*, .*text, ...",
                            )
                            format_dropdown = gr.Dropdown(
                                label="Prompt Format",
                                info="The format you want to apply to final prompt",
                                choices=list(TIPO_DEFAULT_FORMAT.keys()) + ["custom"],
                                value="Both, tag first (recommend)",
                            )
                            ignore_first_n_tags_slider = gr.Slider(
                                label="Ignore First N Tags",
                                minimum=0,
                                maximum=90,
                                step=1,
                                value=0,
                                info="Number of tags to ignore from the beginning of the prompt.",
                            )
                            format_textarea = gr.TextArea(
                                value=TIPO_DEFAULT_FORMAT[
                                    "Both, tag first (recommend)"
                                ],
                                label="Custom Prompt Format",
                                visible=False,
                                placeholder="<|extended|>. <|general|>",
                            )
                            format_dropdown.change(
                                lambda x: gr.update(
                                    visible=x == "custom",
                                    value=TIPO_DEFAULT_FORMAT.get(
                                        x, list(TIPO_DEFAULT_FORMAT.values())[0]
                                    ),
                                ),
                                inputs=format_dropdown,
                                outputs=format_textarea,
                            )

                            with gr.Group():
                                with gr.Row():
                                    seed_num_input = gr.Number(
                                        label="Seed for upsampling tags",
                                        minimum=-1,
                                        maximum=2**31 - 1,
                                        step=1,
                                        scale=4,
                                        value=-1,
                                    )
                                    seed_random_btn = gr.Button(value="Randomize")
                                    seed_shuffle_btn = gr.Button(value="Shuffle")

                                    seed_random_btn.click(
                                        lambda: -1, outputs=[seed_num_input]
                                    )
                                    seed_shuffle_btn.click(
                                        lambda: random.randint(0, 2**31 - 1),
                                        outputs=[seed_num_input],
                                    )

                            with gr.Group():
                                process_timing_dropdown = gr.Dropdown(
                                    label="Upsampling timing",
                                    choices=list(PROCESSING_TIMING.values()),
                                    value=PROCESSING_TIMING["AFTER"],
                                )

                                process_timing_md = gr.Markdown(
                                    on_process_timing_dropdown_changed(
                                        process_timing_dropdown.value
                                    )
                                )

                                process_timing_dropdown.change(
                                    on_process_timing_dropdown_changed,
                                    inputs=[process_timing_dropdown],
                                    outputs=[process_timing_md],
                                )

                        with gr.Column():
                            gr.Markdown(RECOMMEND_MARKDOWN)
                            model_dropdown = gr.Dropdown(
                                label="Model",
                                choices=MODEL_NAME_LIST,
                                value=MODEL_NAME_LIST[0],
                            )
                            gguf_use_cpu = gr.Checkbox(label="Use CPU (GGUF)")
                            no_formatting = gr.Checkbox(
                                label="No formatting", value=False
                            )
                            temperature_slider = gr.Slider(
                                label="Temperature",
                                info="← less random | more random →",
                                maximum=1.5,
                                minimum=0.1,
                                step=0.05,
                                value=0.5,
                            )
                            top_p_slider = gr.Slider(
                                label="Top-p",
                                info="← less unconfident tokens | more unconfident tokens →",
                                maximum=1,
                                minimum=0,
                                step=0.05,
                                value=0.95,
                            )
                            top_k_slider = gr.Slider(
                                label="Top-k",
                                info="← less unconfident tokens | more unconfident tokens →",
                                maximum=150,
                                minimum=0,
                                step=1,
                                value=80,
                            )

        aspect_ratio_place_holder = gr.Number(value=1.0, visible=False)

        prompt_gen.click(
            self.prompt_gen_only,
            inputs=[
                self.tag_prompt_area[is_img2img],
                self.prompt_area[is_img2img * 2 + 1],
                aspect_ratio_place_holder,
                ignore_first_n_tags_slider,
                seed_num_input,
                tag_length_radio,
                nl_length_radio,
                ban_tags_textbox,
                format_dropdown,
                format_textarea,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                model_dropdown,
                gguf_use_cpu,
                no_formatting,
                self.tag_prompt_area[is_img2img],
            ],
            outputs=[
                self.prompt_area[is_img2img * 2],
            ],
        )

        self.infotext_fields = [
            (
                (tipo_acc, lambda d: gr.update(value=INFOTEXT_KEY in d))
                if InputAccordion
                else (tipo_acc, lambda d: gr.update(open=INFOTEXT_KEY in d))
            ),
            (
                self.prompt_area[is_img2img * 2],
                lambda d: d.get(INFOTEXT_KEY_PROMPT, d["Prompt"]),
            ),
            (
                self.prompt_area[is_img2img * 2 + 1],
                lambda d: d.get(INFOTEXT_NL_PROMPT, ""),
            ),
            (orig_prompt_area, lambda d: d["Prompt"]),
            (enabled_check, lambda d: INFOTEXT_KEY in d),
            (ignore_first_n_tags_slider, lambda d: self.get_infotext(d, "ignore_first_n_tags", 0)),
            (seed_num_input, lambda d: self.get_infotext(d, "seed", None)),
            (tag_length_radio, lambda d: self.get_infotext(d, "tag_length", None)),
            (nl_length_radio, lambda d: self.get_infotext(d, "nl_length", None)),
            (ban_tags_textbox, lambda d: self.get_infotext(d, "ban_tags", None)),
            (format_dropdown, lambda d: self.get_infotext(d, "format", None)),
            (format_textarea, lambda d: d.get(INFOTEXT_KEY_FORMAT, None)),
            (
                process_timing_dropdown,
                lambda d: PROCESSING_TIMING.get(
                    self.get_infotext(d, "timing", None), None
                ),
            ),
            (temperature_slider, lambda d: self.get_infotext(d, "temperature", None)),
            (top_p_slider, lambda d: self.get_infotext(d, "top_p", None)),
            (top_k_slider, lambda d: self.get_infotext(d, "top_k", None)),
            (
                model_dropdown,
                lambda d: self.get_infotext(d, "model", None),
            ),
            (gguf_use_cpu, lambda d: self.get_infotext(d, "gguf_cpu", None)),
            (no_formatting, lambda d: self.get_infotext(d, "no_formatting", None)),
        ]

        return [
            enabled_check,
            process_timing_dropdown,
            ignore_first_n_tags_slider,
            seed_num_input,
            tag_length_radio,
            nl_length_radio,
            ban_tags_textbox,
            format_dropdown,
            format_textarea,
            temperature_slider,
            top_p_slider,
            top_k_slider,
            model_dropdown,
            gguf_use_cpu,
            no_formatting,
            self.tag_prompt_area[is_img2img],
            self.prompt_area[is_img2img * 2 + 1],
        ]

    def get_infotext(self, d, target, default):
        return d.get(INFOTEXT_KEY, {}).get(target, default)

    def write_infotext(
        self,
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        prompt: str,
        process_timing: str,
        seed: int,
        *args, # ignore_first_n_tags is now args[0]
    ):
        p.extra_generation_params[INFOTEXT_KEY] = json.dumps(
            {
                "seed": seed,
                "timing": process_timing,
                "ignore_first_n_tags": args[0],
                "tag_length": args[1],
                "nl_length": args[2],
                "ban_tags": args[3],
                "format_selected": args[4],
                "format": args[5],
                "temperature": args[6],
                "top_p": args[7],
                "top_k": args[8],
                "model": args[9],
                "gguf_cpu": args[10],
                "no_formatting": args[11],
            },
            ensure_ascii=False,
        ).translate(QUOTESWAP)
        p.extra_generation_params[INFOTEXT_KEY_PROMPT] = prompt.strip() or args[-1] # tag_prompt is last
        p.extra_generation_params[INFOTEXT_NL_PROMPT] = args[-2] # nl_prompt is second to last
        # args[4] is format_selected
        if args[4] != DEFAULT_FORMAT: # Check format_selected for default
            p.extra_generation_params[INFOTEXT_KEY_FORMAT] = args[5] # format is args[5]

    def process(
        self,
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        is_enabled: bool,
        process_timing: str,
        seed: int,
        *args,
    ):
        """This method will be called after sd-dynamic-prompts and the styles are applied."""

        if not is_enabled:
            return

        if process_timing != PROCESSING_TIMING["AFTER"]:
            return

        self.original_prompt = p.all_prompts
        self.original_hr_prompt = getattr(p, "all_hr_prompts", None)
        aspect_ratio = p.width / p.height
        if seed == -1:
            seed = random.randrange(2**31 - 1)
        seed = int(seed)

        args = list(args)
        if args[3] != "custom":
            args[4] = TIPO_DEFAULT_FORMAT.get(args[3], args[4])

        self.write_infotext(p, p.prompt, "AFTER", seed, *args) # args here includes ignore_first_n_tags as args[0]

        args_list = list(args)
        ignore_first_n_tags_val = args_list.pop(0) # Extract ignore_first_n_tags
        nl_prompt = args_list.pop() # nl_prompt is the last of the original *args from ui()
        # Remaining args_list are: tag_length, nl_length, ..., no_formatting, tag_prompt_area

        new_all_prompts = []
        for prompt, sub_seed in zip(p.all_prompts, p.all_seeds):
            new_all_prompts.append(
                self._process(prompt, nl_prompt, aspect_ratio, ignore_first_n_tags_val, seed + sub_seed, *args_list)
            )

        hr_fix_enabled = getattr(p, "enable_hr", False)

        if hr_fix_enabled:
            if p.hr_prompt != p.prompt:
                new_hr_prompts = []
                for prompt, hr_prompt in zip(p.all_prompts, p.all_hr_prompts):
                    if prompt == hr_prompt:
                        new_hr_prompts.append(prompt)
                    else:
                        new_hr_prompts.append(hr_prompt)
                p.all_hr_prompts = new_hr_prompts
            else:
                p.all_hr_prompts = new_all_prompts
        p.all_prompts = new_all_prompts

    def before_process(
        self,
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        is_enabled: bool,
        process_timing: str,
        seed: int,
        *args,
    ):
        """This method will be called before sd-dynamic-prompts and the styles are applied."""

        if not is_enabled:
            return

        if process_timing != PROCESSING_TIMING["BEFORE"]:
            return

        self.original_prompt = p.prompt
        self.original_hr_prompt = p.hr_prompt
        aspect_ratio = p.width / p.height
        if seed == -1:
            seed = random.randrange(4294967294)
        self.write_infotext(p, p.prompt, "BEFORE", seed, *args) # args here includes ignore_first_n_tags as args[0]
        seed = int(seed + p.seed)

        args_list = list(args)
        ignore_first_n_tags_val = args_list.pop(0) # Extract ignore_first_n_tags
        nl_prompt = args_list.pop() # nl_prompt is the last of the original *args from ui()
        # Remaining args_list are: tag_length, nl_length, ..., no_formatting, tag_prompt_area

        p.prompt = self._process(p.prompt, nl_prompt, aspect_ratio, ignore_first_n_tags_val, seed, *args_list)

    def prompt_gen_only(self, tag_prompt_area, prompt_area, aspect_ratio_place_holder, ignore_first_n_tags_slider_val, seed_num_input_val, *args):
        # Construct the arguments list for _process carefully
        # Original expected args for _process before ignore_first_n_tags:
        # prompt (from tag_prompt_area or prompt_area), nl_prompt (from prompt_area), aspect_ratio, seed, ... other_args
        # The *args at the end of prompt_gen_only's signature will capture the rest of the UI elements
        # The prompt comes from self.prompt_area[is_img2img * 2] which is not directly passed here.
        # It seems prompt_gen_only is used to populate self.prompt_area[is_img2img * 2]
        # The actual prompt that _process uses is derived from its first argument `prompt` which is `tag_prompt_area` (or the main prompt if that's empty)

        # The arguments for _process are:
        # prompt, nl_prompt, aspect_ratio, ignore_first_n_tags, seed, tag_length, nl_length, ban_tags,
        # format_select, format, temperature, top_p, top_k, model, gguf_use_cpu, no_formatting, tag_prompt

        # Mapping inputs from prompt_gen.click to _process parameters:
        # self.tag_prompt_area[is_img2img] -> prompt (becomes first arg to _process)
        # self.prompt_area[is_img2img * 2 + 1] -> nl_prompt (becomes second arg to _process)
        # aspect_ratio_place_holder -> aspect_ratio (becomes third arg to _process)
        # ignore_first_n_tags_slider -> NEW ignore_first_n_tags (will be fourth)
        # seed_num_input -> seed (will be fifth)
        # *args from prompt_gen_only will then follow:
        # tag_length_radio, nl_length_radio, ban_tags_textbox, format_dropdown, format_textarea,
        # temperature_slider, top_p_slider, top_k_slider, model_dropdown, gguf_use_cpu, no_formatting,
        # self.tag_prompt_area[is_img2img] (again, as tag_prompt for _process)

        current_seed = seed_num_input_val
        if current_seed == -1:
            current_seed = random.randrange(2**31 - 1)

        # Reconstruct args for _process in the correct order
        process_args = [
            tag_prompt_area,  # prompt
            prompt_area,      # nl_prompt
            aspect_ratio_place_holder, # aspect_ratio
            ignore_first_n_tags_slider_val, # ignore_first_n_tags
            current_seed,     # seed
        ]
        process_args.extend(args) # The rest of the arguments

        return self._process(*process_args)

    def _process(
        self,
        prompt: str,
        nl_prompt: str,
        aspect_ratio: float,
        ignore_first_n_tags: int,
        seed: int,
        tag_length: str,
        nl_length: str,
        ban_tags: str,
        format_select: str,
        format: str,
        temperature: float,
        top_p: float,
        top_k: int,
        model: str,
        gguf_use_cpu: bool,
        no_formatting: bool,
        tag_prompt: str,
    ):
        prompt = prompt.strip() or tag_prompt

        # Implement the tag ignoring logic
        if ignore_first_n_tags > 0:
            tags_list = [t.strip() for t in prompt.split(',') if t.strip()]
            if ignore_first_n_tags < len(tags_list):
                tags_list = tags_list[ignore_first_n_tags:]
                prompt = ", ".join(tags_list)
            elif ignore_first_n_tags >= len(tags_list) and len(tags_list) > 0 : # if try to ignore all or more than all, leave 1 tag to avoid empty prompt if possible
                prompt = tags_list[-1] # Keep the last tag
            # If tags_list is empty, prompt remains as is (empty or original tag_prompt)

        seed = int(seed) % SEED_MAX
        if model != self.current_model:
            if " | " in model:
                model_name, gguf_name = model.split(" | ")
                target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
                if str(models.model_dir / target_file) not in models.list_gguf():
                    models.download_gguf(model_name, gguf_name)
                target = os.path.join(str(models.model_dir), target_file)
                gguf = True
            else:
                target = model
                gguf = False
            models.load_model(target, gguf, device="cpu" if gguf_use_cpu else "cuda")
            self.current_model = model
        prompt_preview = prompt.replace("\n", " ")[:40]
        logger.info(f"Processing prompt: {prompt_preview}...")
        logger.info(f"Processing with seed: {seed}")
        prompt_without_extranet, res = parse_prompt(prompt)
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

        nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
        nl_prompt = ""
        strength_map_nl = []
        for part, strength in nl_prompt_parse_strength:
            nl_prompt += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))

        rebuild_extranet = ""
        for name, params in res.items():
            for param in params:
                items = ":".join(param.items)
                rebuild_extranet += f" <{name}:{items}>"

        black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
        tipo.BAN_TAGS = black_list
        all_tags = []
        strength_map = {}
        break_map = set()
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            if part == "BREAK" and strength == -1:
                break_map.add(all_tags[-1])
                continue
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength

        tag_length = tag_length.replace(" ", "_")
        nl_length = nl_length.replace(" ", "_")
        org_tag_map = seperate_tags(all_tags)

        meta, operations, general, nl_prompt = parse_tipo_request(
            org_tag_map,
            nl_prompt,
            tag_length_target=tag_length,
            nl_length_target=nl_length,
            generate_extra_nl_prompt=(not nl_prompt and "<|extended|>" in format)
            or "<|generated|>" in format,
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

        if isinstance(models.text_model, torch.nn.Module):
            models.text_model.to(devices.device)
        tag_map, _ = tipo_runner(
            meta,
            operations,
            general,
            nl_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        if isinstance(models.text_model, torch.nn.Module):
            models.text_model.cpu()
            devices.torch_gc()

        addon = {
            "tags": [],
            "nl": "",
        }
        for cate in tag_map.keys():
            if cate == "generated" and addon["nl"] == "":
                addon["nl"] = tag_map[cate]
                continue
            if cate == "extended":
                extended = tag_map[cate]
                addon["nl"] = extended
                continue
            if cate not in org_tag_map:
                continue
            for tag in tag_map[cate]:
                if tag in org_tag_map[cate]:
                    continue
                addon["tags"].append(tag)
        addon = apply_strength(addon, strength_map, strength_map_nl, break_map)
        unformatted_prompt_by_tipo = (
            prompt + ", " + ", ".join(addon["tags"]) + "\n" + addon["nl"]
        )
        tag_map = apply_strength(tag_map, strength_map, strength_map_nl, break_map)
        formatted_prompt_by_tipo = apply_format(tag_map, format).replace(
            "BREAK,", "BREAK"
        )

        if no_formatting:
            final_prompt = unformatted_prompt_by_tipo
        else:
            final_prompt = formatted_prompt_by_tipo

        result = final_prompt + "\n" + rebuild_extranet
        logger.info("Prompt processing done.")
        return result


def parse_infotext(_, params):
    try:
        params[INFOTEXT_KEY] = json.loads(params[INFOTEXT_KEY].translate(QUOTESWAP))
    except Exception:
        pass


scripts.script_callbacks.on_infotext_pasted(parse_infotext)

options.categories.register_category("prompt_gen", "Prompt Gen")
shared.options_templates.update(
    shared.options_section(
        ("TIPO", "TIPO", "prompt_gen"),
        {
            "tipo_no_extra_input": shared.OptionInfo(
                False,
                (
                    "Disable extra input for TIPO"
                    ", Natural Language Prompt and Tag Prompt will be hidden."
                    " (UI Reload Needed)"
                ),
            ),
        },
    )
)
