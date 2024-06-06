import gradio as gr
import matplotlib.pyplot as plt
import torch
import os
# from demo_utils import call,  DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV, LoRANetwork
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from GD_Utils import *
from diffusers.pipelines import StableDiffusionXLPipeline

StableDiffusionXLPipeline.__call__ = call

model_map = {
    'Brightness Visual' : 'models/brightness-visualslider.pt',
    'Flare Visual' : 'models/flare-visualslider.pt',
    'Age Text':'models/age-textslider.pt' ,
    'Brightness Text': 'models/brightness-textslider.pt',
    # Paper Models
    # 'Age': 'models/age.pt',
    # 'Chubby': 'models/chubby.pt',
    # 'Muscular': 'models/muscular.pt',
    # 'Surprised Look': 'models/suprised_look.pt',
    # 'Smiling': 'models/smiling.pt',
    # 'Professional': 'models/professional.pt',
    # 'Long Hair': 'models/long_hair.pt',
    # 'Curly Hair': 'models/curlyhair.pt',

    # 'Pixar Style': 'models/pixar_style.pt',
    # 'Sculpture Style': 'models/sculpture_style.pt',
    # 'Clay Style': 'models/clay_style.pt',

    # 'Repair Images': 'models/repair_slider.pt',
    # 'Fix Hands': 'models/fix_hands.pt',

    # 'Cluttered Room': 'models/cluttered_room.pt',

    # 'Dark Weather': 'models/dark_weather.pt',
    # 'Festive': 'models/festive.pt',
    # 'Tropical Weather': 'models/tropical_weather.pt',
    # 'Winter Weather': 'models/winter_weather.pt',

    # 'Wavy Eyebrows': 'models/eyebrow.pt',
    # 'Small Eyes (use scales -3, -1, 1, 3)': 'models/eyesize.pt',
}



class Demo:

    def __init__(self) -> None:

        self.generating = False
        self.device = 'cuda'
        self.weight_dtype = torch.float16

        # # Use SDXL Normal
        # model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
        # pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=self.weight_dtype).to(self.device)
        # pipe = None
        # del pipe
        # torch.cuda.empty_cache()

        model_id = "stabilityai/sdxl-turbo"
        self.current_model = 'SDXL Turbo'
        euler_anc = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=euler_anc,
                                                              torch_dtype=self.weight_dtype).to(self.device)

        self.guidance_scale = 1
        self.num_inference_steps = 3
        self.generated_images = []
        self.prompts = set()

        with gr.Blocks() as demo:
            self.layout()
            demo.queue(max_size=5).launch(share=True, max_threads=2, debug=True)

    def layout(self):

        with gr.Row():
            with gr.Tab("Inference") as inference_column:
                with gr.Row():
                    self.explain_infr = gr.Markdown(
                        value='This is our demo of the paper Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models. Pick a LoRa Concept slider and input a promt. Use the slider scale to adjust the strength of the Concept.  </b>')

                with gr.Row():
                    with gr.Column(scale=1):
                        self.prompt_input_infr = gr.Text(
                            placeholder="photo of a flash light, camera glare, flare, reflective, realistic, 8k",
                            label="Prompt",
                            info="Prompt to generate",
                            value="photo flash light, camera glare, flare, reflective, realistic, 8k"
                        )

                        with gr.Row():
                            self.model_dropdown = gr.Dropdown(
                                label="Pretrained Sliders",
                                choices=list(model_map.keys()),
                                value='Flare',
                                interactive=True
                            )

                            self.seed_infr = gr.Number(
                                label="Seed",
                                value=21
                            )

                            self.slider_scale_infr = gr.Slider(
                                -10,
                                10,
                                label="Slider Scale",
                                value=0,
                                info="Larger slider scale result in stronger edit",
                                step=1
                            )

                            self.slider_scale_infsteps = gr.Slider(
                                0,
                                50,
                                label="Inference steps",
                                value=self.num_inference_steps,
                                info="Amount of inference steps",
                                step=1
                            )

                            self.start_noise_infr = gr.Slider(
                                600, 900,
                                value=600,
                                label="SDEdit Timestep",
                                info="Choose smaller values for more structural preservation"
                            )

                    with gr.Column(scale=2):
                        self.infr_button = gr.Button(
                            value="Generate",
                            interactive=True
                        )

                    # with gr.Column(scale=2):
                    #     self.result_button = gr.Button(
                    #         value='Generate Results',
                    #         interactive=True
                    #     )

                        with gr.Row():
                            self.image_orig = gr.Image(
                                label="Original SD",
                                interactive=False,
                                type='pil',
                            )

                            self.image_new = gr.Image(
                                label=f"Concept Slider",
                                interactive=False,
                                type='pil',
                            )

            with gr.Tab("Images") as image_column:
                    with gr.Row():
                        self.explain_infr = gr.Markdown(
                            value='This is our demo of the paper Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models. Pick a LoRa Concept slider and input a promt. Use the slider scale to adjust the strength of the Concept.  </b>')
                    with gr.Row():
                      with gr.Column(scale=2):
                        self.glr_button = gr.Button(
                            value="Clear_Gallery",
                            interactive=True
                        )
                        self.crt_button = gr.Button(
                            value="Update_Galery",
                            interactive=True
                        )
                    gallery = gr.Gallery(label="Generated images", show_label=True, elem_id="gallery",
                                         columns=[4], rows=[1], object_fit="contain", height="auto")

        self.glr_button.click(self.clear_galery, inputs=None,
                              outputs=gallery)
        self.crt_button.click(self.update_galary, inputs=None,
                              outputs=gallery)

        self.infr_button.click(self.inference, inputs=[
            self.prompt_input_infr,
            self.seed_infr,
            self.slider_scale_infsteps,
            self.start_noise_infr,
            self.slider_scale_infr,
            self.model_dropdown,
        ], outputs=[
               self.image_new,
               self.image_orig]
           )

        # self.result_button.click(self.inference_results, inputs=[
        #     self.prompt_input_infr,
        #     self.seed_infr,
        #     self.slider_scale_infsteps,
        #     self.start_noise_infr,
        #     self.slider_scale_infr,
        #     self.model_dropdown,
        # ], outputs=None)


    def clear_galery(self):
      images = []
      self.prompts = set()
      self.generated_images = images
      return images

    def update_galary(self):
      return self.generated_images

    def inference_results(self, prompt, seed, inf_steps, start_noise, scale, model_name, model, pbar=gr.Progress(track_tqdm=True)):
      image_list = []
      scales = [-2, 1, 0, 1, 2]
      for scale in scales:
        image = self.inference(
            prompt, seed, inf_steps, start_noise, scale, model_name, model)
        image_list.append(image[0])
      fig, ax = plt.subplots(1, len(image_list), figsize=(20,4))
      for i, a in enumerate(ax):
          a.imshow(image_list[i])
          a.set_title(f"{scales[i]}",fontsize=15)
          a.axis('off')

      plt.suptitle(f'{model_name}', fontsize=20)
      # plt.tight_layout()
      plt.show()
      plt.savefig(f'images/{prompt}{seed}.png')

    def inference(self, prompt, seed, inf_steps, start_noise, scale, model_name, model, pbar=gr.Progress(track_tqdm=True)):
        seed = seed or 420
        if self.current_model != model:
            if model == 'SDXL Turbo':
                model_id = "stabilityai/sdxl-turbo"
                euler_anc = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=euler_anc,
                                                                      torch_dtype=self.weight_dtype).to(self.device)
                self.guidance_scale = 1
                self.num_inference_steps = inf_steps
                self.current_model = 'SDXL Turbo'
            # else:
            #     model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
            #     self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=self.weight_dtype).to(
            #         self.device)
            #     self.pipe.enable_xformers_memory_efficient_attention()
            #     self.guidance_scale = 7.5
            #     self.num_inference_steps = 20
            #     self.current_model = 'SDXL'
        generator = torch.manual_seed(seed)
        self.num_inference_steps = inf_steps
        model_path = model_map[model_name]
        unet = self.pipe.unet
        network_type = "c3lier"
        if 'full' in model_path:
            train_method = 'full'
        elif 'noxattn' in model_path:
            train_method = 'noxattn'
        elif 'xattn' in model_path:
            train_method = 'xattn'
            network_type = 'lierla'
        else:
            train_method = 'noxattn'

        modules = DEFAULT_TARGET_REPLACE
        if network_type == "c3lier":
            modules += UNET_TARGET_REPLACE_MODULE_CONV

        name = os.path.basename(model_path)
        rank = 4
        alpha = 1
        if 'rank' in model_path:
            rank = int(float(model_path.split('_')[-1].replace('.pt', '')))
        if 'alpha1' in model_path:
            alpha = 1.0
        network = LoRANetwork(
            unet,
            rank=rank,
            multiplier=1.0,
            alpha=alpha,
            train_method=train_method,
        ).to(self.device, dtype=self.weight_dtype)
        network.load_state_dict(torch.load(model_path))

        generator = torch.manual_seed(seed)
        edited_image = \
        self.pipe(prompt, num_images_per_prompt=1, num_inference_steps=self.num_inference_steps, generator=generator,
                  network=network, start_noise=int(start_noise), scale=float(scale), unet=unet,
                  guidance_scale=self.guidance_scale).images[0]

        generator = torch.manual_seed(seed)
        original_image = \
        self.pipe(prompt, num_images_per_prompt=1, num_inference_steps=self.num_inference_steps, generator=generator,
                  network=network, start_noise=start_noise, scale=0, unet=unet,
                  guidance_scale=self.guidance_scale).images[0]

        del unet, network
        unet = None
        network = None
        torch.cuda.empty_cache()
        if prompt not in self.prompts:
          self.generated_images.append((original_image, f"[original] {prompt}"))
          self.prompts.add(prompt)
        self.generated_images.append((edited_image, f"[edited scale: {scale}] {prompt}"))

        return edited_image, original_image


demo = Demo()

