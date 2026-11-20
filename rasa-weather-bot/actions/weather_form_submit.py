import os

import requests
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# 心知天气 API Key 从环境变量读取，不要硬编码到代码里
SENIVERSE_API_KEY = os.getenv("SENIVERSE_API_KEY", "")


def get_weather(location, date):
    match date:
        case "今天":
            index_of_date = 0
        case "明天":
            index_of_date = 1
        case "后天":
            index_of_date = 2
        case _:
            return "暂时只支持3天内的天气查询"
    url = "https://api.seniverse.com/v3/weather/daily.json"
    result = requests.get(
        url,
        params={
            "key": SENIVERSE_API_KEY,
            "location": location,
        },
        timeout=2,
    )
    result = result.json()["results"][0]["daily"]
    return f"""{location} {date}的天气：
白天：{result[index_of_date]["text_day"]}
夜晚：{result[index_of_date]["text_night"]}
最高温度：{result[index_of_date]["high"]}℃
最低温度：{result[index_of_date]["low"]}℃
降水概率：{result[index_of_date]["precip"]}
风向：{result[index_of_date]["wind_direction"]}
风力：{result[index_of_date]["wind_scale"]}级
湿度：{result[index_of_date]["humidity"]}%
    """


class WeatherFormSubmit(Action):
    def name(self) -> Text:
        return "action_weather_form_submit"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        location = tracker.get_slot("location")
        date = tracker.get_slot("date")
        message = "服务暂时故障，请稍后再试"
        try:
            message = get_weather(location, date)
        except Exception:
            pass
        dispatcher.utter_message(message)
        return []
