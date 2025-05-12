from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path
import base64, time


class LLM:

    from openai import APITimeoutError

    class models:
        gemini_2_0_flash = "gemini-2.0-flash"
        gemini_2_5_pro_exp_03_25 = "gemini-2.5-pro-exp-03-25"
        gemini_2_0_flash_thinking = "gemini-2.0-flash-thinking-exp-01-21"
        gemini_2_0_flash_lite = "gemini-2.0-flash-lite"
        deepseek_deepseek_chat_v3_0324_nitro = "deepseek/deepseek-chat-v3-0324:nitro"
        deepseek_deepseek_chat_v3_0324_free = "deepseek/deepseek-chat-v3-0324:free"
        deepseek_deepseek_r1 = "deepseek/deepseek-r1"
        deepseek_deepseek_r1_nitro = "deepseek/deepseek-r1:nitro"
        TA_deepseek_ai_deepseek_r1 = "TA/deepseek-ai/DeepSeek-R1"
        ARK_deepseek_v3_250324 = "ark-deepseek-v3-250324"
        doubao_pro_32k_241215 = "doubao-pro-32k-241215"
        doubao_vision_pro_32k_241028 = "doubao-vision-pro-32k-241028"
        doubao_vision_lite_32k_241015 = "doubao-vision-lite-32k-241015"
        hunyuan_t1_latest = "hunyuan-t1-latest"
        hunyuan_turbos_latest = "hunyuan-turbos-latest"
        qwen2_vl_72b = "Qwen/Qwen2-VL-72B-Instruct"
        chatgpt_4o_latest = "chatgpt-4o-latest"
        o3_mini = "o3-mini"
        bge_m3 = "BAAI/bge-m3"

    def __init__(
        self,
        model=models.gemini_2_0_flash,
        base_url="https://oneapi.jyj.cx/v1",
        key=f"sk-jiangyj",
        timeout=900,
        max_retries=2,
        log_to=None,
    ):
        self.base_url = base_url
        self.key = key
        self.chat_model = model
        self.embed_model = LLM.models.bge_m3
        self.timeout = timeout
        self.max_retries = max_retries
        self.log_file = log_to

    def client(self):
        return OpenAI(
            api_key=self.key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,  # 已实现自定义超时处理，内置超时重试存在指数级回退情况（弃用）
        )

    def chat(
        self,
        prompt: str,
        context: str | None = None,
        image: str | list[str] | None = None,
        schema: BaseModel | None = None,
        model: str | None = None,
        output: list | None = None,
        statistics: dict | None = None,
        silent: bool = False,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        # 共享输出
        if output is None:
            output = []
        elif output != []:
            raise ValueError("output must be an empty list")
        output.append("")

        if statistics is None:
            statistics = {}
        elif statistics != {}:
            raise ValueError("statistics must be an empty dict")

        prompt: list = [{"type": "text", "text": prompt}]

        def encode_image(img_path):
            if Path(img_path).exists():
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                    return f"data:image/jpeg;base64,{img_b64}"
            raise ValueError(f"Image file not found: {img_path}")

        if image:
            images = image if isinstance(image, list) else [image]
            for img in images:
                if encoded := encode_image(img):
                    prompt.append({"type": "image_url", "image_url": {"url": encoded}})

        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ]
        messages.pop(0) if context is None else None

        model = model if model else self.chat_model

        timeout = timeout if timeout else self.timeout
        max_retries = max_retries if max_retries else self.max_retries

        for attempt in range(max_retries + 1):
            try:
                if schema is None:
                    # 无特定响应格式的流式返回
                    return self._chat_stream(
                        messages=messages,
                        model=model,
                        output=output,
                        statistics=statistics,
                        slient=silent,
                        timeout=timeout,
                    )
                else:
                    # 特定响应格式的返回（JSON 模式）
                    return self._chat_schema(
                        messages=messages,
                        schema=schema,
                        model=model,
                        statistics=statistics,
                        slient=silent,
                        timeout=timeout,
                    )

            except LLM.APITimeoutError as e:
                self._handle_api_error(e, attempt, max_retries)

            except Exception as e:
                self._handle_api_error(e, attempt, max_retries)

    def _chat_stream(
        self,
        messages: list,
        model: str,
        output: list,
        statistics: dict,
        slient: bool,
        timeout: float,
    ):
        t = time.time()

        if self.log_file:
            self._log_basic_info(model, prompt=messages[-1]["content"])

        stream = self.client().chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            timeout=timeout,
        )

        first_token = True
        for chunk in stream:
            if chunk.choices:
                # 检查是否超时
                if time.time() - t > timeout:
                    # 关闭流式响应
                    try:
                        stream.close()
                    except:
                        pass
                    print("\n[LLM] 响应返回超时，已关闭请求！")
                    raise TimeoutError

                if (delta := chunk.choices[0].delta.content) is not None:
                    output[0] += delta
                    if first_token:
                        first_token = False
                        statistics["ttft"] = round(time.time() - t, 3)
                    if not slient:
                        print(delta, end="", flush=True)
                    if self.log_file:
                        with open(self.log_file, "a", encoding="utf-8") as f:
                            f.write(delta)

        statistics["total_time"] = round(time.time() - t, 3)
        if chunk.usage:
            statistics["prompt_tokens"] = chunk.usage.prompt_tokens
            statistics["completion_tokens"] = chunk.usage.completion_tokens
            statistics["total_tokens"] = chunk.usage.total_tokens
            statistics["tps"] = int(
                round(statistics["total_tokens"] / statistics["total_time"], 0)
            )

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write("\n\n")

        return output[0]

    def _chat_schema(
        self,
        messages: list,
        schema: BaseModel,
        model: str,
        slient: bool,
        timeout: float,
    ):
        if self.log_file:
            self._log_basic_info(model, messages[-1]["content"])

        completion = self.client().beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
            timeout=timeout,
        )

        if completion.choices[0].message.parsed:
            response: BaseModel = completion.choices[0].message.parsed
        else:
            response: str = completion.choices[0].message.refusal

        # 写入响应内容到日志
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(str(response))
                f.write("\n")

        if not slient:
            print(response)

        return response

    def embed(self, text: str, model: str | None = None) -> list[float]:
        model = model if model is not None else self.embed_model
        response = self.client().embeddings.create(input=text, model=model)
        return response.data[0].embedding

    def _handle_api_error(self, e, attempt, max_retries):
        if attempt >= max_retries:
            raise e

        if isinstance(e, LLM.APITimeoutError):
            print(
                f"[{self.chat_model}] Request timeout, retrying ({attempt + 1}/{max_retries}) ..."
            )
