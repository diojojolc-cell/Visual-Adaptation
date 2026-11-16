"""Attach B4 to a single Qwen3-VL Generate call, during prefill only."""

from contextlib import AbstractContextManager

from .memory import VisualMemory


class B4Inference(AbstractContextManager):
    """Use one context per full, unpadded prompt with a fresh generation cache."""

    def __init__(self, model, adapter, inputs, question_option_positions):
        self.model = model
        self.adapter = adapter
        self.inputs = inputs
        self.question_option_positions = question_option_positions
        self.core = model.model.language_model
        self.memory = None
        self.keys = None
        self.prefill = False
        self.handles = []

    def __enter__(self):
        self.handles.append(self.core.register_forward_pre_hook(
            self._begin, with_kwargs=True))
        for layer in self.adapter.layers:
            self.handles.append(self.core.layers[layer].mlp.register_forward_hook(
                self._writer(layer)))
        return self

    def _begin(self, module, args, kwargs):
        self.prefill = self.memory is None
        if not self.prefill:
            return
        embeds = kwargs['inputs_embeds']
        ids = self.inputs['input_ids'].to(embeds.device)
        attention_mask = self.inputs['attention_mask'].to(embeds.device)
        rope_positions, _ = self.model.model.get_rope_index(
            ids, image_grid_thw=self.inputs.get('image_grid_thw'),
            video_grid_thw=self.inputs.get('video_grid_thw'),
            attention_mask=attention_mask)
        self.memory = VisualMemory.from_prompt(
            input_ids=ids, inputs_embeds=embeds, attention_mask=attention_mask,
            rope_positions=rope_positions,
            question_option_positions=self.question_option_positions,
            image_token_id=self.model.config.image_token_id,
            video_token_id=self.model.config.video_token_id)
        self.keys = self.adapter.keys(self.memory)

    def _writer(self, layer):
        def inject(module, args, native_output):
            if not self.prefill:
                return native_output
            active = self.memory.active
            state = args[0][0, active]
            delta = self.adapter.output_delta(layer, state, self.memory, self.keys)
            output = native_output.clone()
            output[0, active] = native_output[0, active] + delta.to(native_output.dtype)
            return output
        return inject

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.memory = self.keys = None
        self.prefill = False
        return False
