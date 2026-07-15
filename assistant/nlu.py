import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """你是一个智能语音助手的意图理解引擎。你的任务是从用户的语音输入中识别意图并提取关键实体。

请严格以 JSON 格式输出，不要包含其他内容：
{
    "intent": "意图名称",
    "confidence": 0.0-1.0,
    "entities": {"实体名": "实体值"},
    "response": "对用户的回复"
}

支持的意图：

1. open_app - 打开应用程序
   实体: app_name (应用名称)
   示例: "打开计算器" → {"intent": "open_app", "entities": {"app_name": "计算器"}}

2. search - 搜索信息
   实体: query (搜索关键词)
   示例: "搜索今天的新闻" → {"intent": "search", "entities": {"query": "今天的新闻"}}

3. time_query - 查询时间或日期
   无需实体
   示例: "现在几点了" → {"intent": "time_query", "entities": {}}

4. weather - 查询天气
   实体: location (地点，可省略)
   示例: "今天天气怎么样" → {"intent": "weather", "entities": {}}

5. system_cmd - 系统命令
   实体: command (shutdown, sleep, volume_up, volume_down, mute, screenshot)
   示例: "关机" → {"intent": "system_cmd", "entities": {"command": "shutdown"}}
   示例: "静音" → {"intent": "system_cmd", "entities": {"command": "mute"}}

6. chat - 自由对话（没有匹配到以上意图时使用）
   示例: "讲个笑话" → {"intent": "chat", "entities": {}, "response": "..."}

注意事项：
- confidence 表示你对意图判断的确信程度，0-1之间
- 对于 chat 意图，response 字段填你的回复内容
- 对于其他意图，response 字段告知用户你在执行什么操作
- 所有回复请使用中文
- 如果用户的话可能有多种理解，选择最可能的那个"""


class NLUEngine:
    def __init__(self, config):
        self.config = config
        self._client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self._conversation_history = []

    def understand(self, text):
        if not text:
            return None

        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        ]

        for msg in self._conversation_history[-6:]:
            messages.append(msg)

        messages.append({"role": "user", "content": text})

        try:
            response = self._client.chat.completions.create(
                model=self.config.DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            result = json.loads(content)
            logger.info(f"NLU: [{result.get('intent')}] ({result.get('confidence', 0):.2f}) {text}")

            self._conversation_history.append({"role": "user", "content": text})
            self._conversation_history.append({"role": "assistant", "content": content})

            return result

        except json.JSONDecodeError:
            logger.warning(f"NLU JSON parse failed, treating as chat intent: {content}")
            return {
                "intent": "chat",
                "confidence": 0.5,
                "entities": {},
                "response": content,
            }
        except Exception as e:
            logger.error(f"NLU error: {e}")
            return {
                "intent": "chat",
                "confidence": 0.0,
                "entities": {},
                "response": "抱歉，我暂时无法理解你的意思。",
            }

    def reset_conversation(self):
        self._conversation_history = []
